#
# 부동산 PDF 문서를 활용한 RAG 시스템 구축 예제
# DATA 폴더에 부동산 PDF 파일들을 넣어주세요.
#

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS  # Chroma 대신 FAISS 로드


# 변경 코드
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

# 1. 초기 설정 및 API 키 입력
st.set_page_config(page_title="Brisbane Real Estate AI Agent", layout="wide")
st.title("🏡 Brisbane Real Estate AI Q & A")
st.caption("From Kevin Oh")

# 1. Properly read from Streamlit Secrets dashboard
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # Fallback for your local VS Code environment
    api_key = os.getenv("GEMINI_API_KEY")

#  api_key = os.getenv("GEMINI_API_KEY")
# 1. API 키 설정 (Google AI Studio에서 발급받은 키 입력)

with st.sidebar:
    # st.header("Settings")
    # api_key = st.text_input("Enter your Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.info("""
    **학습 대상 자료:**
    - 부동산 PDF 자료
    - 정부 부동산 관련 PDF 자료 DATA directory 기준.
   
    """)

# 2. 문서 로드 및 FAISS 벡터 스토어 생성/로드 함수
@st.cache_resource
def setup_rag_system(data_folder):
    if not os.path.exists(data_folder):
        return None
        
    faiss_index_path = "./faiss_db"
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # 기존 코드 수정
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 이미 만들어진 FAISS 인덱스가 있다면 로드 (중복 연산 방지)
    if os.path.exists(faiss_index_path):
        try:
            # allow_dangerous_deserialization=True는 로컬에서 안전하게 저장된 pickle 파일을 읽을 때 필수적입니다.
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            st.warning(f"기존 FAISS 로드 실패, 다시 생성합니다: {e}")

    all_docs = []
    # 폴더 내 모든 PDF 파일 로드
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
    
    if not all_docs:
        return None

    # 텍스트 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(all_docs)

    # FAISS 벡터 저장소 생성 및 로컬 저장
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore

# 3. 데이터 로딩 및 시스템 구축
data_path = "data" 
if not os.path.exists(data_path):
    os.makedirs(data_path)
    st.warning(f"'{data_path}' 폴더에 부동산 자료 PDF 파일들을 넣어주세요.")

vectorstore = None
if api_key:
    with st.spinner("부동산 법률 문서를 분석 중입니다 (FAISS)..."):
        vectorstore = setup_rag_system(data_path)

# 4. 커스텀 프롬프트 템플릿 설정 (Prompt Engineering)
custom_prompt_template = """너는 호주 퀸즐랜드(QLD) 주 부동산법에 정통한 수석 변호사 보조이자 최고 수준의 부동산 전문가 AI 에이전트야.
사용자의 질문에 대해 아래에 제공된 문맥(Context) 내용만을 엄격하게 바탕으로 정확하고 전문적으로 답변해야 해.

[답변 지침]
1. 부동산 법적 규정,  설명할 때는 관련 조항이나 문서의 근거를 명확히 제시해.
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
    # 대화 체인 생성 (gemini-2.5-pro 적용)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False
    )

    # 채팅 인터페이스
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("호주 부동산에 대해 물어보세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("자료를 기반으로 답변을 작성 중입니다..."):
                response = qa_chain.invoke({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    if not api_key:
        st.info("시작하려면 사이드바에 Google API Key를 입력하세요.")
    elif not vectorstore:
        st.error(f"'{data_path}' 폴더에 PDF 자료가 없습니다. 자료를 넣고 페이지를 새로고침 하세요.")

# 면책 조항
# st.markdown("---")
# st.caption("⚠️ 면책 조항: 본 AI 에이전트의 답변은 법적 조언이 아니며, 참고용입니다. 공식적인 법률 판단은 반드시 변호사 또는 REIQ 전문가와 상의하십시오.")