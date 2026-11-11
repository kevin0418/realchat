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
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "file_manifest" not in st.session_state:
    st.session_state.file_manifest = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_result" not in st.session_state:
    st.session_state.summary_result = ""
if "detected_domain" not in st.session_state:
    st.session_state.detected_domain = "일반"

# --- 429 처리: 재시도 래퍼 함수 ---
def call_gemini_with_retry(model: str, contents: list, max_retries: int = 3):
    """429 Resource Exhausted 발생 시 안내 메시지와 함께 대기 후 자동 재시도"""
    status_placeholder = st.empty()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            status_placeholder.empty()
            return response.text
        except ClientError as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_sec = 60  # 기본 대기 60초
                status_placeholder.warning(
                    f"⚠️ Gemini 1분당 토큰 한도(100만 토큰)에 도달했습니다. "
                    f"잠시 대기 후 자동으로 재시도합니다... (시도 {attempt + 1}/{max_retries}, 대기: {wait_sec}초)"
                )
                time.sleep(wait_sec)
            else:
                status_placeholder.empty()
                raise e
        except Exception as e:
            status_placeholder.empty()
            raise e
            
    status_placeholder.error("❌ 재시도 횟수를 초과했습니다. 파일 크기나 개수를 줄여 다시 시도해주세요.")
    return None

# --- 이미지 최적화 함수 (토큰 대폭 절감) ---
def resize_image_for_gemini(img_bytes: bytes, max_dimension: int = 1024) -> bytes:
    """고해상도 도면/사진을 리사이즈 및 JPEG 압축하여 전송 토큰과 용량을 최적화"""
    try:
        image = Image.open(io.BytesIO(img_bytes))
        # RGBA -> RGB 변환
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # 비율 유지 축소
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        out_io = io.BytesIO()
        image.save(out_io, format="JPEG", quality=80, optimize=True)
        return out_io.getvalue()
    except Exception:
        return img_bytes  # 실패 시 원본 유지

# --- 단일 파일 스트림 파싱 함수 ---
def parse_txt_stream(file_stream) -> str:
    try:
        content = file_stream.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    except Exception as e:
        return f"[TXT 읽기 오류: {e}]"

def parse_docx_stream(file_stream) -> str:
    try:
        doc = Document(file_stream)
        content = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_text = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                table_data.append(" | ".join(row_text))
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

# --- 도메인 자동 판별 함수 ---
def detect_data_domain(text_corpus: str) -> str:
    snippet = text_corpus[:6000].lower()
    real_estate_kw = ["zoning", "overlay", "council", "lease", "tenant", "land", "dwelling", "lot", "boundary", "property", "부동산", "임대", "조닝", "계약", "토지", "건물", "카운슬"]
    religion_kw = ["god", "jesus", "lord", "church", "pastor", "bible", "prayer", "faith", "gospel", "하나님", "예수", "교회", "목사", "성경", "기도", "은혜", "말씀", "신앙", "설교"]
    health_kw = ["patient", "doctor", "hospital", "clinic", "symptom", "treatment", "medicine", "diagnosis", "health", "환자", "병원", "진료", "증상", "치료", "약품", "건강", "의사", "혈압", "혈당"]

    re_score = sum(snippet.count(w) for w in real_estate_kw)
    rel_score = sum(snippet.count(w) for w in religion_kw)
    health_score = sum(snippet.count(w) for w in health_kw)

    scores = {"부동산 (Real Estate)": re_score, "종교/신앙 (Religion)": rel_score, "건강/의료 (Health)": health_score}
    top_domain, top_score = max(scores.items(), key=lambda x: x[1])
    return top_domain if top_score >= 3 else "일반 종합 문서 (General)"

def get_domain_persona(domain: str) -> str:
    if "부동산" in domain:
        return (
            "당신은 부동산, 도시계획(Planning Scheme), 상업용 리스 및 개발 규정 전문 컨설턴트입니다. "
            "조닝(Zoning), 오버레이(Overlay), 부지 면적(Site Cover), 임대차 조건, 수치적 규제 조건을 매우 엄격하고 정밀하게 검토하여 답변하세요."
        )
    elif "종교" in domain:
        return (
            "당신은 깊이 있는 영적 통찰과 신학적 문맥을 이해하는 종교/신앙 연구자입니다. "
            "성경 구절 인용, 말씀의 본래 배경, 교훈적 의미 및 실천적 묵상 포인트를 따뜻하고 격조 높은 어조로 정리하세요."
        )
    elif "건강" in domain:
        return (
            "당신은 의학 지식과 건강 가이드를 객관적으로 정리해주는 건강 분석 비서입니다. "
            "임상 수치, 복약 주의사항, 생활 습관 가이드라인을 명확하게 설명하되, 전문적인 의료 상담을 대체할 수 없다는 점을 균형 있게 짚어주세요."
        )
    return "당신은 제공된 문서를 객관적으로 분석하여 질문에 충실하게 답하는 전문 지식 분석가입니다."

# --- 파일 목록 통합 처리기 ---
def process_file_items(file_items):
    manifest = []
    text_corpus = []
    image_parts = []

    for item in file_items:
        name = item["name"]
        ext = os.path.splitext(name)[1].lower()
        stream = item["stream"]

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
                # 고해상도 이미지는 압축 및 리사이즈 적용
                optimized_bytes = resize_image_for_gemini(raw_bytes, max_dimension=1024)
                part = types.Part.from_bytes(data=optimized_bytes, mime_type="image/jpeg")
                image_parts.append({"name": name, "part": part})
                manifest.append({"name": name, "type": ext, "info": f"이미지 ({len(optimized_bytes)//1024} KB로 압축됨)"})
            except Exception as e:
                st.warning(f"이미지 {name} 처리 실패: {e}")
            continue
        else:
            continue

        if extracted.strip():
            manifest.append({"name": name, "type": ext, "info": f"{len(extracted):,}자"})
            text_corpus.append(
                f"=====================================================\n"
                f"DOCUMENT: {name}\n"
                f"TYPE: {ext}\n"
                f"=====================================================\n"
                f"{extracted}\n\n"
            )

    full_text_str = "".join(text_corpus)
    combined_contents = []
    if full_text_str:
        combined_contents.append(full_text_str)
    for img in image_parts:
        combined_contents.append(f"\n[IMAGE ATTACHMENT: {img['name']}]\n")
        combined_contents.append(img["part"])

    detected = detect_data_domain(full_text_str)
    return combined_contents, manifest, detected

# --- 사이드바 ---
with st.sidebar:
    st.header("📂 입력 방식 선택")
    input_mode = st.radio("자료 공급 방식", ["로컬 폴더 경로 지정", "개별 파일 직접 업로드"])
    
    ready_items = []
    
    if input_mode == "로컬 폴더 경로 지정":
        folder_path = st.text_input("디렉터리 경로", value="./property_data")
        if st.button("폴더 내 파일 일괄 읽기", use_container_width=True):
            if os.path.isdir(folder_path):
                valid_exts = {".txt", ".md", ".docx", ".xlsx", ".xls", ".pdf", ".jpg", ".jpeg", ".png"}
                for root, _, files in os.walk(folder_path):
                    for f in sorted(files):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_exts:
                            fp = os.path.join(root, f)
                            with open(fp, "rb") as fh:
                                ready_items.append({"name": f, "stream": io.BytesIO(fh.read())})
            else:
                st.error("올바른 디렉터리 경로를 입력하세요.")
    else:
        uploaded_files = st.file_uploader(
            "파일들을 선택하거나 끌어다 놓으세요",
            type=["txt", "md", "docx", "xlsx", "xls", "pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if st.button("업로드된 파일 처리 시작", use_container_width=True):
            if uploaded_files:
                for uf in uploaded_files:
                    ready_items.append({"name": uf.name, "stream": io.BytesIO(uf.read())})
            else:
                st.warning("업로드된 파일이 없습니다.")

    if ready_items:
        with st.spinner("자료 파싱, 이미지 최적화 및 토큰 계산 중..."):
            contents, manifest, domain = process_file_items(ready_items)
            if manifest:
                st.session_state.processed_docs = contents
                st.session_state.file_manifest = manifest
                st.session_state.detected_domain = domain
                st.session_state.chat_history = []
                st.session_state.summary_result = ""
                
                # 사전 토큰 수 측정
                try:
                    tok_resp = client.models.count_tokens(model="gemini-2.5-flash", contents=contents)
                    total_tokens = tok_resp.total_tokens
                    if total_tokens > 950000:
                        st.error(f"⚠️ 현재 로드된 토큰이 약 {total_tokens:,}개로 1분당 한도(100만)에 매우 근접했습니다! 파일을 분할해 분석하세요.")
                    else:
                        st.success(f"{len(manifest)}개 파일 로드 완료! (예상 토큰: {total_tokens:,} / 한도: 1,000,000)")
                except Exception:
                    st.success(f"{len(manifest)}개 파일 로드 완료!")
            else:
                st.warning("유효한 텍스트 또는 이미지 내용을 찾을 수 없습니다.")

    if st.session_state.file_manifest:
        st.divider()
        st.markdown(f"**감지된 자료 성격**: `{st.session_state.detected_domain}`")
        st.subheader("로드된 파일 목록")
        for f in st.session_state.file_manifest:
            st.caption(f"• **{f['name']}** ({f['type']}) - {f['info']}")

# --- 메인 화면 ---
st.title("📑 스마트 문서/자료 맞춤 분석기")

if not st.session_state.processed_docs:
    st.info("왼쪽 사이드바에서 [폴더 경로]를 입력하거나 [파일들]을 직접 업로드하여 시작하세요.")
    st.stop()

st.caption(f"현재 분석 관점: **{st.session_state.detected_domain} 특화 엔진**")

tab1, tab2 = st.tabs(["📊 맞춤 종합 요약", "💬 심층 질의응답 (Q&A)"])

# [탭 1: 종합 요약]
with tab1:
    if st.button("자료 성격에 맞춘 종합 요약 생성", type="primary"):
        persona = get_domain_persona(st.session_state.detected_domain)
        prompt = [
            persona,
            *st.session_state.processed_docs,
            f"""
            위 자료 전체를 검토하여 다음 양식에 맞추어 종합 보고서를 작성해주세요:
            1. 자료의 주된 목적 및 핵심 배경
            2. [{st.session_state.detected_domain}] 관점에서 가장 중요한 핵심 내용 및 주요 수치/조항/메시지
            3. 문서 및 이미지별 세부 요약과 참조 출처(파일명 명시)
            4. 향후 검토나 주의가 필요한 사항
            """
        ]
        with st.spinner("요약 생성 중 (토큰 한도 초과 시 자동 대기 후 재시도)..."):
            resp_text = call_gemini_with_retry(model="gemini-2.5-flash", contents=prompt)
            if resp_text:
                st.session_state.summary_result = resp_text

    if st.session_state.summary_result:
        st.markdown(st.session_state.summary_result)
        st.download_button(
            label="요약 보고서 다운로드 (.md)",
            data=st.session_state.summary_result,
            file_name=f"summary_{st.session_state.detected_domain.split()[0]}.md",
            mime="text/markdown"
        )

# [탭 2: Q&A 질의응답]
with tab2:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    if query := st.chat_input("문서 내용에 관해 질문하세요 (예: 특정 조항, 수치, 교훈, 증상 등)"):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        persona = get_domain_persona(st.session_state.detected_domain)
        qa_instruction = (
            f"{persona}\n"
            "반드시 제공된 문서 및 이미지 내용에만 근거하여 답변하세요. "
            "추측은 배제하고 각 핵심 답변마다 [출처: 파일명]을 명시하세요. 근거가 없으면 명시적으로 찾을 수 없다고 답변하세요."
        )
        qa_contents = [
            qa_instruction,
            *st.session_state.processed_docs,
            f"\n\n질문: {query}"
        ]
        
        with st.chat_message("assistant"):
            with st.spinner("답변 작성 중..."):
                ans = call_gemini_with_retry(model="gemini-2.5-flash", contents=qa_contents)
                if ans:
                    st.markdown(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})

# --- 사이드바 하단: Q&A 로그 다운로드 ---
if st.session_state.chat_history:
    with st.sidebar:
        st.divider()
        st.write("### 💾 질의응답 기록 저장")
        log_text = [f"=== 문서 분석 Q&A 로그 (분야: {st.session_state.detected_domain}) ===\n\n"]
        for c in st.session_state.chat_history:
            speaker = "사용자" if c["role"] == "user" else "Gemini 답변"
            log_text.append(f"[{speaker}]\n{c['content']}\n\n" + "-"*40 + "\n\n")
        
        st.download_button(
            label="Q&A 대화 내역 다운로드 (.txt)",
            data="".join(log_text),
            file_name=f"qa_log_{st.session_state.detected_domain.split()[0]}.txt",
            mime="text/plain",
            use_container_width=True
        )