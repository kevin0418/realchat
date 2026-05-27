#
# 부동산 PDF, word  문서를 upload 하여 RAG 시스템 구축 예제
# 

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 1. 초기 설정 및 API 키 입력
st.set_page_config(page_title="호주 부동산 질문 by upload files", layout="wide")
st.title("🏡 호주 부동산 질문 by upload files")
st.caption("From Kevin Oh")

if "api_keys" in st.secrets and "GEMINI_API_KEY" in st.secrets["api_keys"]:
    api_key = st.secrets["api_keys"]["GEMINI_API_KEY"]
else:
    # 2. st.secrets에 없다면 Windows 환경변수에서 가져옴
    api_key = os.getenv("GEMINI_API_KEY")

# api_key = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.info("""
    **학습 대상 권장 자료:**
    - 호주 부동산 정부 자료
    - 부동산 자료 모음
    - 부동산 뉴스 
    """)
    
    st.markdown("---")
    st.header("📄 문서 업로드")
    # PDF와 Word 파일 동시 수락 (다중 업로드 허용)
    uploaded_files = st.file_uploader(
        "분석할 법률 PDF 또는 Word 파일을 선택하세요.", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

# 2. 업로드된 파일들을 로드 및 처리하는 함수
@st.cache_resource
def setup_rag_from_uploads(files):
    if not files:
        return None
        
    all_docs = []
    
    # LangChain 로더는 파일 경로를 요구하므로 임시(Temp) 파일 생성 기법을 사용합니다.
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_file_path)
                
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다 ({uploaded_file.name}): {e}")
        finally:
            # 임시 파일 사용 후 디스크 청소
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    if not all_docs:
        return None

    # 텍스트 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(all_docs)

    # 매번 업로드 세션에 최적화된 새로운 FAISS 인덱스 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# 3. 데이터 로딩 및 시스템 구축
vectorstore = None
if api_key and uploaded_files:
    with st.spinner("업로드된 부동산 문서를 텍스트 추출 및 분석 중입니다 (FAISS)..."):
        vectorstore = setup_rag_from_uploads(uploaded_files)

# 4. 커스텀 프롬프트 템플릿 설정 (Prompt Engineering)
custom_prompt_template = """너는 호주 퀸즐랜드(QLD) 주 부동산법에 정통한 수석 변호사 보조이자 최고 수준의 부동산 전문가 AI 에이전트야.
사용자의 질문에 대해 아래에 제공된 문맥(Context) 내용만을 엄격하게 바탕으로 정확하고 전문적으로 답변해야 해.

[답변 지침]
1. 법적 규정, 설명 때는 관련 조항이나 문서의 근거를 명확히 제시해.
2. 답변은 항상 정중하고 전문적인 어조를 유지하고, 호주 부동산 전문 용어는 괄호 안에 영어를 병기해. (예: 위임장(Form 6), 수수료(Commission))
3. 답변을 구조화하여 읽기 쉽게 글머리 기호를 사용해.
4. **가장 중요:** 제공된 문맥에 질문에 대한 답이 없다면, 절대 내용을 지어내지 말고 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 정직하게 답변하여 법적 오류(Hallucination)를 방지해.

문맥 (Context):
{context}

사용자 질문: {question}

전문가의 답변:"""

QA_PROMPT = PromptTemplate(
    template=custom_prompt_template, 
    input_variables=["context", "question"]
)

# 5. 대화 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

# 6. 질문 답변 로직
if vectorstore and api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False
    )

    # 채팅 인터페이스 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("호주 부동산 관련 자료, 또는 정부 규정에 대해 물어보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("부동산 자료를 기반으로 답변을 작성 중입니다..."):
                response = qa_chain.invoke({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    if not api_key:
        st.info("시작하려면 시스템 환경변수에 Google API Key를 등록하거나 코드를 확인하세요.")
    elif not uploaded_files:
        st.warning("👈 왼쪽 사이드바에서 분석할 PDF 또는 Word(.docx) 문서를 먼저 업로드해 주세요.")

# 면책 조항
# st.markdown("---")
# st.caption("⚠️ 면책 조항: 본 AI 에이전트의 답변은 법적 조언이 아니며, 참고용입니다. 공식적인 법률 판단은 반드시 변호사 또는 REIQ 전문가와 상의하십시오.")
