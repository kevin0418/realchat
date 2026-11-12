import os
import io
import time
import streamlit as st
import pandas as pd
from PIL import Image
from docx import Document
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import ClientError

st.set_page_config(page_title="스마트 다중 포맷 문서 분석 & Q&A", layout="wide")

# 1. API 키 설정
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 또는 Streamlit secrets를 확인해주세요.")
    st.stop()

client = genai.Client(api_key=api_key)

# 세션 상태 초기화
if "active_contents" not in st.session_state:
    st.session_state.active_contents = []
if "loaded_count" not in st.session_state:
    st.session_state.loaded_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_result" not in st.session_state:
    st.session_state.summary_result = ""

# --- 429 처리 및 호출 함수 ---
def call_gemini_with_retry(model: str, system_instruction: str, contents: list, max_retries: int = 3):
    status_placeholder = st.empty()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3
    )
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            status_placeholder.empty()
            return response.text
        except ClientError as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_sec = 60
                status_placeholder.warning(
                    f"⚠️ Gemini 1분당 입력 토큰 한도(100만 토큰)에 도달했습니다. "
                    f"잠시 대기 후 자동으로 재시도합니다... (시도 {attempt + 1}/{max_retries}, 대기: {wait_sec}초)"
                )
                time.sleep(wait_sec)
            else:
                status_placeholder.empty()
                raise e
        except Exception as e:
            status_placeholder.empty()
            raise e
            
    status_placeholder.error("❌ 재시도 횟수를 초과했습니다. 업로드 파일 수를 줄여 다시 시도해주세요.")
    return None

# --- 이미지 최적화 함수 ---
def resize_image_for_gemini(img_bytes: bytes, max_dimension: int = 1024) -> bytes:
    try:
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        out_io = io.BytesIO()
        image.save(out_io, format="JPEG", quality=80, optimize=True)
        return out_io.getvalue()
    except Exception:
        return img_bytes

# --- 단일 파일 스트림 파싱 함수 ---
def parse_txt_stream(file_stream) -> str:
    try:
        content = file_stream.read()
        return content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
    except Exception as e:
        return f"[TXT 읽기 오류: {e}]"

def parse_docx_stream(file_stream) -> str:
    try:
        doc = Document(file_stream)
        content = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            table_data = [" | ".join([cell.text.strip().replace("\n", " ") for cell in row.cells]) for row in table.rows]
            if table_data:
                content.append("\n[TABLE]\n" + "\n".join(table_data) + "\n[/TABLE]\n")
        return "\n".join(content)
    except Exception as e:
        return f"[Word 읽기 오류: {e}]"

def parse_excel_stream(file_stream) -> str:
    try:
        excel_file = pd.ExcelFile(file_stream)
        sheet_texts = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            clean_df = df.dropna(how="all")
            sheet_texts.append(f"--- [Sheet: {sheet_name}] ---\n" + clean_df.to_string(index=False))
        return "\n\n".join(sheet_texts)
    except Exception as e:
        return f"[Excel 읽기 오류: {e}]"

def parse_pdf_stream(file_stream) -> str:
    try:
        reader = PdfReader(file_stream)
        pages = [f"[Page {i+1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
        return "\n\n".join(pages)
    except Exception as e:
        return f"[PDF 읽기 오류: {e}]"

# --- 4대 분야별 전문가 페르소나 및 요약 구조 ---
def get_domain_config(domain: str):
    if domain == "부동산, 법규":
        expert_role = (
            "당신은 호주(퀸즐랜드/브리즈번/로건) 및 일반 부동산 거래, 도시계획(City Plan / Planning Scheme), "
            "조닝(Zoning), 오버레이(Overlay), 건폐율(Site Cover), 높이 제한, 셋백(Setback), 상업용 임대차(Commercial Lease) 및 부동산 법규에 정통한 수석 부동산 컨설턴트입니다. "
            "실무적 규정, 수치, 예외 조건을 정확하고 체계적으로 설명하세요."
        )
        summary_structure = """
        위 자료를 [부동산, 법규] 전문가 관점에서 다음 4가지 항목으로 요약 보고서를 작성하세요:
        1. **자료의 핵심 주제 및 개요**: 대상 부동산 또는 문서가 다루는 주요 부동산/시장/법규 내용
        2. **주요 수치, 규제 및 핵심 조건**: 문서에 등장하는 가격, 임대료, 면적, 규제 기준, 전망치 등 핵심 데이터 정리
        3. **실무적 시사점 및 주의사항**: 매매·임대·개발 관점에서 주목해야 할 포인트
        4. **문서별 세부 내용 요약**: 업로드된 각 파일별 핵심 요약 및 출처 명시
        """
    elif domain == "종교, 철학":
        expert_role = (
            "당신은 깊이 있는 신학적 통찰, 경전 주해 능력, 동서양 철학 사조에 정통한 인문학 및 영성 연구자입니다. "
            "핵심 교리, 철학적 명제, 경전 구절, 역사적 배경과 실천적 묵상 포인트를 깊이 있게 풀어내세요."
        )
        summary_structure = """
        위 자료를 [종교, 철학] 연구자 관점에서 다음 4가지 항목으로 깊이 있게 요약하세요:
        1. **중심 주제 및 핵심 명제**: 자료 전체를 관통하는 본질적인 신학적·철학적 메시지
        2. **주요 논지 전개 및 경전/원전 근거**: 핵심 논거의 흐름과 인용된 말씀/사상가/배경 정리
        3. **영적 통찰 및 삶에의 적용점**: 내면적 성찰과 실천적 교훈
        4. **자료별 세부 내용 요약**: 각 문서별 고유 내용과 출처 표기
        """
    elif domain == "건강, 의학":
        expert_role = (
            "당신은 임상 데이터, 건강검진 결과, 의학 논문 및 헬스케어 가이드를 객관적이고 체계적으로 분석하는 의료 정보 분석 전문가입니다. "
            "검사 수치, 정상 참고치, 약물 정보, 이상 소견을 알기 쉽게 설명하세요. (의학적 조언 시 전문의 진료 권고 포함)"
        )
        summary_structure = """
        위 자료를 [건강, 의학] 분석 전문가 관점에서 다음 4가지 항목으로 명확하게 정리하세요:
        1. **핵심 건강 상태 및 주요 소견 요약**: 전반적인 개요와 가장 주목해야 할 핵심 결과
        2. **주요 검사 수치 및 기준치 비교**: 정상 범위 내 항목과 주의/이상 수치 대조
        3. **처방/치료 계획 및 주의사항**: 복약 지도, 경고사항, 생활 습관 관리 지침
        4. **자료별 세부 기록 요약**: 각 검사지/문서별 핵심 내용과 출처 명시
        """
    else:  # 기타
        expert_role = (
            "당신은 방대한 텍스트, 표(Excel/Word), 시각 자료(이미지)에서 핵심 인사이트를 도출하고 "
            "폭넓은 배경지식을 바탕으로 명확한 해답을 제시하는 수석 지식 분석가입니다."
        )
        summary_structure = """
        위 자료 전체를 검토하여 다음 4가지 항목으로 체계적인 종합 보고서를 작성하세요:
        1. **자료의 핵심 목적 및 전체 개요**: 문서들이 다루는 주요 주제와 배경
        2. **가장 중요한 핵심 내용 및 주요 수치/일정**: 반드시 파악해야 할 핵심 팩트 정리
        3. **파일별 세부 요약**: 업로드된 각 문서 및 이미지별 주요 내용과 출처 명시
        4. **시사점 및 후속 검토 필요 사항**: 자료를 바탕으로 도출되는 결론 및 추가 확인 포인트
        """
    return expert_role, summary_structure

# --- 업로드 파일 파싱 처리기 ---
def parse_uploaded_files(uploaded_files):
    combined_parts = []
    valid_count = 0
    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()
        stream = io.BytesIO(uf.read())

        if ext in [".txt", ".md"]:
            extracted = parse_txt_stream(stream)
        elif ext == ".docx":
            extracted = parse_docx_stream(stream)
        elif ext in [".xlsx", ".xls"]:
            extracted = parse_excel_stream(stream)
        elif ext == ".pdf":
            extracted = parse_pdf_stream(stream)
        elif ext in [".jpg", ".jpeg", ".png"]:
            try:
                raw_bytes = stream.read()
                optimized_bytes = resize_image_for_gemini(raw_bytes, max_dimension=1024)
                part = types.Part.from_bytes(data=optimized_bytes, mime_type="image/jpeg")
                combined_parts.append(f"\n[IMAGE ATTACHMENT: {name}]\n")
                combined_parts.append(part)
                valid_count += 1
            except Exception as e:
                st.warning(f"이미지 {name} 처리 실패: {e}")
            continue
        else:
            continue

        if extracted.strip():
            formatted_text = (
                f"=====================================================\n"
                f"DOCUMENT: {name}\n"
                f"TYPE: {ext}\n"
                f"=====================================================\n"
                f"{extracted}\n\n"
            )
            combined_parts.append(formatted_text)
            valid_count += 1

    return combined_parts, valid_count

# --- 2단계 분리형 Q&A 엔진 ---
def generate_hybrid_answer(domain: str, active_contents: list, query: str) -> str:
    expert_role, _ = get_domain_config(domain)
    
    # [1단계] 첨부된 문서 내에서만 답변 가능 여부 판별 및 추출
    doc_check_sys = (
        "당신은 엄격한 문서 검색기입니다. 제공된 문서 안에 사용자 질문에 대한 직접적인 답이나 관련 정보가 있는지 확인하세요.\n"
        "- 만약 관련 정보가 있다면 첫 줄에 반드시 'FOUND: YES'라고 쓰고, 이어서 문서 내용에 근거한 상세 답변과 [출처: 파일명]을 작성하세요.\n"
        "- 만약 제공된 문서가 전혀 다른 주제의 문서이거나 질문에 답할 내용이 없다면, 다른 말은 절대 쓰지 말고 오직 'FOUND: NO' 한 단어만 출력하세요."
    )
    doc_check_contents = [
        *active_contents,
        f"\n\n[질문]: {query}"
    ]
    
    step1_result = call_gemini_with_retry(
        model="gemini-2.5-flash",
        system_instruction=doc_check_sys,
        contents=doc_check_contents
    )
    
    if not step1_result:
        return "응답을 생성하지 못했습니다. 다시 시도해주세요."

    # [2단계 분기]
    if step1_result.strip().startswith("FOUND: YES"):
        clean_doc_answer = step1_result.replace("FOUND: YES", "", 1).strip()
        return f"### 📄 제공된 자료 기반 답변\n\n{clean_doc_answer}"
    else:
        ai_fallback_sys = (
            f"{expert_role}\n"
            "사용자의 질문에 대해 당신이 보유한 전문 지식을 총동원하여 명확하고 체계적이며 실무적으로 도움이 되는 상세한 답변을 작성하세요."
        )
        ai_fallback_contents = [
            f"다음 질문에 대해 [{domain}] 분야 전문가로서 상세하고 정확하게 답변해주세요:\n\n질문: {query}"
        ]
        
        step2_result = call_gemini_with_retry(
            model="gemini-2.5-flash",
            system_instruction=ai_fallback_sys,
            contents=ai_fallback_contents
        )
        
        return (
            "ℹ️ *제공된 자료에서는 해당 질문에 대한 직접적인 내용을 찾을 수 없어, AI 전문 지식으로 답변을 생성했습니다.*\n\n"
            f"### 🤖 AI 답변 ({domain} 전문 지식 기반)\n\n{step2_result}"
        )

# --- 사이드바 ---
with st.sidebar:
    st.header("1️⃣ 분석 분야 선택")
    selected_domain = st.radio(
        "자료 성격 및 질문 분야를 선택하세요:",
        options=["부동산, 법규", "종교, 철학", "건강, 의학", "기타"],
        index=0
    )

    st.divider()
    st.header("2️⃣ 분석할 파일 업로드")
    uploaded_files = st.file_uploader(
        "Word, Excel, PDF, TXT, JPG 파일을 선택하거나 끌어다 놓으세요",
        type=["txt", "md", "docx", "xlsx", "xls", "pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("업로드 파일 분석 준비", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("파일 파싱 및 이미지 최적화 중..."):
                parts, count = parse_uploaded_files(uploaded_files)
                if count > 0:
                    st.session_state.active_contents = parts
                    st.session_state.loaded_count = count
                    st.session_state.chat_history = []
                    st.session_state.summary_result = ""
                    st.success(f"총 {count}개 파일 준비 완료!")
                else:
                    st.warning("유효한 텍스트나 이미지 내용을 추출할 수 있는 파일이 없습니다.")
        else:
            st.warning("먼저 파일을 업로드해주세요.")

# --- 메인 화면 ---
st.title("📑 문서 분석 요약 질문 by Kevin")

if not st.session_state.active_contents:
    st.info("👈 왼쪽 사이드바에서 **[1. 분석 분야]**를 고르고 **[2. 파일 업로드]** 후 **[업로드 파일 분석 준비]** 버튼을 눌러주세요.")
    st.stop()

st.caption(f"현재 적용된 전문 분석 엔진: **[{selected_domain}] 특화 모드** (로드된 파일: {st.session_state.loaded_count}개)")

tab1, tab2 = st.tabs(["📊 맞춤 종합 요약", "💬 심층 질의응답 (Q&A)"])

# [탭 1: 종합 요약]
with tab1:
    if st.button(f"[{selected_domain}] 관점으로 종합 요약 생성", type="primary"):
        expert_role, summary_guide = get_domain_config(selected_domain)
        prompt_contents = [
            *st.session_state.active_contents,
            f"\n\n{summary_guide}"
        ]
        with st.spinner(f"[{selected_domain}] 전문가 관점으로 정밀 요약 보고서를 작성 중입니다..."):
            resp_text = call_gemini_with_retry(
                model="gemini-2.5-flash",
                system_instruction=expert_role,
                contents=prompt_contents
            )
            if resp_text:
                st.session_state.summary_result = resp_text

    if st.session_state.summary_result:
        st.markdown(st.session_state.summary_result)
        st.download_button(
            label="요약 보고서 다운로드 (.md)",
            data=st.session_state.summary_result,
            file_name=f"summary_{selected_domain.replace(', ', '_')}.md",
            mime="text/markdown"
        )

# [탭 2: Q&A 질의응답]
with tab2:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    if query := st.chat_input(f"[{selected_domain}] 자료 내용이나 관련 지식을 자유롭게 질문하세요..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("자료 검색 및 AI 전문 지식 분석 중..."):
                ans = generate_hybrid_answer(
                    domain=selected_domain,
                    active_contents=st.session_state.active_contents,
                    query=query
                )
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

# --- 사이드바 하단: Q&A 로그 다운로드 ---
if st.session_state.chat_history:
    with st.sidebar:
        st.divider()
        st.write("### 💾 질의응답 기록 저장")
        log_text = [f"=== 문서 분석 Q&A 로그 (분야: {selected_domain}) ===\n\n"]
        for c in st.session_state.chat_history:
            speaker = "사용자" if c["role"] == "user" else "Gemini 답변"
            log_text.append(f"[{speaker}]\n{c['content']}\n\n" + "-"*40 + "\n\n")
        
        st.download_button(
            label="Q&A 대화 내역 다운로드 (.txt)",
            data="".join(log_text),
            file_name=f"qa_log_{selected_domain.replace(', ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )