import os
import time
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage 

# Streamlit 페이지 설정 (가장 상단에 위치)
st.set_page_config(page_title="메모리네비 💬📚", page_icon="🧭", layout="centered")

# OpenAI API 키 설정
# 환경 변수 MY_OPENAI_API_KEY가 설정되어 있지 않으면 오류 메시지 출력 후 앱 중단
openai_api_key = os.getenv("MY_OPENAI_API_KEY")
if openai_api_key is None:
    st.error("오류: OpenAI API 키가 설정되지 않았습니다. 'MY_OPENAI_API_KEY' 환경 변수를 설정해주세요.")
    st.stop() # API 키가 없으면 앱 실행을 중단합니다.
os.environ["OPENAI_API_KEY"] = openai_api_key

# @st.cache_resource: 한 번 실행된 결과를 캐싱하여 재실행 시 시간을 절약
@st.cache_resource
def load_and_split_pdf(file_path):
    """PDF 파일을 로드하고 페이지별로 분할합니다."""
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(_docs):
    """문서 청크를 FAISS 벡터스토어에 임베딩하여 저장합니다."""
    # 텍스트 분할기 설정: 청크 크기 500, 중복 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)

    # OpenAI 임베딩 모델 설정: text-embedding-3-small, 차원 1536 명시 (FAISS 차원 불일치 방지)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 로컬에 FAISS 인덱스 저장
    vectorstore.save_local("faiss_index")
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    """로컬에 저장된 FAISS 벡터스토어를 로드하거나, 없으면 새로 생성합니다."""
    faiss_index_path = "faiss_index"
    # 인덱스 파일이 존재하면 로드
    if os.path.exists(os.path.join(faiss_index_path, "index.faiss")) and \
       os.path.exists(os.path.join(faiss_index_path, "index.pkl")):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    # 없으면 새로 생성
    else:
        return create_vector_store(_docs)


@st.cache_resource
def initialize_components(selected_model):
    """
    data 폴더의 모든 PDF를 로드하여 벡터 DB를 만들고,
    검색 및 채팅 히스토리를 포함한 전체 LangChain 체인을 구성합니다.
    """
    data_dir = "./data"
    all_pages = []

    # 'data' 디렉토리가 없으면 생성하고 사용자에게 알림
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.info(f"'{data_dir}' 디렉토리가 생성되었습니다. PDF 파일을 이 안에 넣어주세요.")
        return None # PDF 파일이 없으므로 컴포넌트 초기화 중단

    pdf_found = False
    # data 디렉토리 내의 모든 PDF 파일을 로드
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            try:
                pages = load_and_split_pdf(file_path)
                all_pages.extend(pages)
                pdf_found = True
            except Exception as e:
                st.warning(f"⚠️ {filename} 불러오기 실패: {e}")

    # 유효한 PDF 문서가 하나도 없으면 에러 메시지 출력
    if not pdf_found:
        st.error(f"❌ '{data_dir}' 폴더에 유효한 PDF 문서가 없습니다.")
        return None

    # 벡터스토어 생성 및 retriever 추출
    vectorstore = get_vectorstore(all_pages)
    retriever = vectorstore.as_retriever()

    # 히스토리 기반 retriever를 위한 시스템 프롬프트 정의
    contextualize_q_system_prompt = """주어진 채팅 히스토리와 최신 사용자 질문을 바탕으로,
        채팅 히스토리 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요.
        질문에 답하지 말고, 필요시 재구성하거나 그대로 반환하세요."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"), # 채팅 히스토리 플레이스홀더
            ("human", "{input}"), # 사용자 입력 플레이스홀더
        ]
    )

    # QA 시스템 프롬프트 정의 (답변 규칙 포함)
    qa_system_prompt = """당신은 주어진 문서를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

    당신의 주요 임무는 치매를 준비하는 노인을 위한 국가 지원 제도, 복지 혜택, 의료 및 돌봄 정보 등 치매 전반에 대한 질문에 대해 정확하고 유익한 정보를 제공하는 것입니다. 
    답변은 간결하고 **최대한 8줄 이내로 설명**해주고, 어린이도 이해할 수 있을 정도로 쉬운 언어로 설명해 주세요.

    --- 
    **답변 기준 및 규칙** 1. 아래에 제공된 문서(context)가 존재할 경우
        - 반드시 **context 내용만을 기반으로** 답변하세요. 
        - 일반적인 배경지식은 사용하지 마세요.
        - 출처나 쪽수는 **표시하지 마세요.**
        - 연락처를 물어보는경우 연락처를 안내해주세요. 
        

    2. 아래 context가 비어 있을 경우
        - GPT 모델이 알고 있는 일반적인 지식만을 사용해 답변하세요. 
        - 이 경우, 반드시 다음 문장을 **답변의 첫머리에 줄바꿈 2번 후 출력**하세요 : 이 답변은 제가 가진 일반적인 정보로 알려 드리는 거예요.

    3. 문서에서 제공되지 않는 정보의 경우 (이 규칙은 2번과 유사하게 작동)
        - **답변의 첫머리에 줄바꿈 2번 후 출력**하세요 : 이 답변은 제가 가진 일반적인 정보로 알려 드리는 거예요.

          --- 
          참고 문서 (context): 
          {context}

      """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"), # 채팅 히스토리 플레이스홀더
            ("human", "{input}"), # 사용자 입력 플레이스홀더
        ]
    )

    # LLM, 히스토리 인식 리트리버, QA 체인 구성
    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 최종 RAG 체인 반환
    return rag_chain


# Streamlit UI 시작
st.markdown("""
<div class="title-section">
    <h1>🧭 메모리네비</h1>
    <p>어르신을 위한 치매 관련 정보 도우미입니다.<br>
    궁금한 내용을 편하게 물어보세요!</p>
</div>
""", unsafe_allow_html=True)

# CSS 스타일 정의
st.markdown("""
<style>
/* 사용자 입력창 글자 크기 */
.stTextInput > div > input {
    font-size: 24px !important;
}

/* 타이틀 영역 스타일 */
.title-section h1 {
    color: #000000 !important;
}
.title-section p {
    color: #555555 !important;
}
.title-section {
    text-align: center;
    background-color: #f8f9f9;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    border: 1px solid #ddd;
}

/* AI 메시지 버블 내부의 텍스트와 <p> 태그 폰트 크기 */
div[data-testid="stChatMessage"][data-variant="assistant"] {
    font-size: 24px !important; /* AI 메시지 버블 전체의 기본 폰트 크기 */
    line-height: 1.8 !important;
}
div[data-testid="stChatMessage"][data-variant="assistant"] p {
    font-size: 24px !important; /* AI 메시지 버블 내부 <p> 태그의 폰트 크기 */
    line-height: 1.8 !important;
}

/* 사용자 메시지 버블 내부의 텍스트와 <p> 태그 폰트 크기 */
div[data-testid="stChatMessage"][data-variant="user"] {
    font-size: 24px !important; /* 사용자 메시지 버블 전체의 기본 폰트 크기 */
    line-height: 1.6 !important;
}
div[data-testid="stChatMessage"][data-variant="user"] p {
    font-size: 24px !important; /* 사용자 메시지 버블 내부 <p> 태그의 폰트 크기 */
    line-height: 1.6 !important;
}
            
/* 참고 문서 확인 폰트 크기 조절 */
.reference-docs-content,
.reference-docs-content p,
.reference-docs-content span,
.reference-docs-content li { /* 리스트 항목도 포함 */
    font-size: 18px !important; /* 참고 문서 폰트 크기 */
    line-height: 1.5 !important;
}

/* 초기 AI 버블(“치매에 대해 무엇이든…”) 전체 높이·여백 확대 */
div[data-testid="stChatMessage"][data-variant="assistant"]:first-of-type {
    padding: 1.2rem 1.5rem !important;  /* 상하·좌우 여백 ↑ */
    min-height: 96px !important;        /* 칸 자체 기본 높이 ↑ */
}
            
/* 입력창(placeholder 포함) */
/* ChatInput 박스 자체 높이 늘리기 */
[data-testid="stChatInput"] > div:first-child {
    min-height: 64px !important;    /* 원하는 높이 (px) */
    display: flex;
    align-items: center;            /* 세로 가운데 정렬 */
    padding: 0 1rem !important;
}
/* 실제 입력 textarea */
[data-testid="stChatInput"] textarea {
    font-size: 20px !important;     /* 입력 글자 크기 */
    line-height: 1.6 !important;
    padding: 0.6rem 0.5rem !important;
}
/* placeholder 글꼴도 동일하게 */
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 20px !important;
    opacity: 0.7;                   /* 살짝 옅게 보이도록 */
}
</style>
""", unsafe_allow_html=True)


# 모델을 gpt-4o-mini로 고정 (selectbox 제거)
selected_model = "gpt-4o-mini"
rag_chain = initialize_components(selected_model) # initialize_components 함수에 고정된 모델 전달

# StreamlitChatMessageHistory 인스턴스 생성
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 세션 상태 초기화 (문맥/채팅 메시지)
if "context" not in st.session_state:
    st.session_state["context"] = []

# 초기 환영 메시지 (히스토리에 없으면 추가)
if not chat_history.messages:
    chat_history.add_ai_message("치매에 대해 무엇이든 물어보세요! 🧠✨")

# 이전 대화 메시지 출력 (히스토리에서만 읽어서 표시)
# 이 루프는 앱이 새로고침될 때마다 모든 메시지를 다시 그립니다.
for msg in chat_history.messages:
    if msg.type == "human":
        with st.chat_message("human"):
            st.markdown(f"<span style='font-size:24px; color:#007BFF;'>{msg.content}</span>", unsafe_allow_html=True)
    else: # msg.type == "ai"
        with st.chat_message("ai"):
            # AI 메시지는 CSS 규칙에 따라 폰트 크기 24px 적용
            st.markdown(f"<span style='font-size:24px;'>{msg.content}</span>", unsafe_allow_html=True)


# rag_chain이 성공적으로 초기화되었을 때만 conversational_rag_chain을 생성
if rag_chain:
    # RunnableWithMessageHistory는 내부적으로 chat_history를 업데이트합니다.
    # 이 객체를 통해 LangChain 체인에 메시지 히스토리를 전달합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history, # 세션 ID에 따라 chat_history 객체 반환
        input_messages_key="input",      # 사용자 입력이 전달될 키
        history_messages_key="history",  # 채팅 히스토리가 전달될 키
        output_messages_key="answer",    # LLM 응답이 반환될 키
    )
else:
    st.info("PDF 문서가 없거나 로드에 실패하여 챗봇을 사용할 수 없습니다. 'data' 폴더에 PDF 파일을 넣어주세요.")
    conversational_rag_chain = None # 챗봇 비활성화


# 사용자가 질문을 입력하는 부분
# st.chat_input은 사용자가 입력을 제출하면 True를 반환하고, prompt_message에 입력값을 할당합니다.
if prompt_message := st.chat_input("치매에 대해 궁금한 점을 여기에 입력해 주세요."):
    if conversational_rag_chain: # 챗봇이 활성화된 경우에만 질문 처리
        # 1. 사용자 메시지를 chat_history에 추가
        #    이 메시지는 다음 st.rerun() 시점에 위 for 루프에 의해 화면에 그려집니다.
        chat_history.add_user_message(prompt_message)
        
        # 2. AI 응답을 위한 빈 플레이스홀더 메시지를 chat_history에 추가
        #    이 메시지 역시 다음 st.rerun() 시점에 빈 AI 버블로 화면에 그려집니다.
        chat_history.add_ai_message("") 
        
        # 3. 앱을 새로고침하여 사용자 메시지와 빈 AI 버블을 즉시 표시
        #    이것이 없으면 AI 응답이 나올 때까지 사용자 메시지가 보이지 않습니다.
        st.rerun() 

# --- 이 부분은 `st.chat_input` 블록 밖에 있어야 합니다. ---
# Streamlit은 사용자 입력 후 전체 스크립트를 다시 실행하므로,
# 이 조건문은 앱이 재실행될 때마다 검사됩니다.
# 이 로직은 마지막 메시지가 '비어있는 AI 메시지'일 때만 실행되어,
# 해당 메시지를 LLM 응답으로 채우는 역할을 합니다.
if chat_history.messages and \
   chat_history.messages[-1].type == "ai" and \
   chat_history.messages[-1].content == "":
    
    # 마지막 메시지가 비어있는 AI 메시지 플레이스홀더라면
    # 이 메시지에 응답을 생성하고 타이핑 효과를 적용합니다.
    with st.chat_message("ai"):
        # `st.empty()`를 사용하여 메시지 내용을 동적으로 업데이트할 플레이스홀더 생성
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("생각 중입니다... 🧐"):
            # LangChain 체인 호출
            # `conversational_rag_chain.invoke`는 내부적으로 `chat_history`를 참조하여
            # 전체 대화 기록을 LLM에 전달합니다.
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke(
                {"input": chat_history.messages[-2].content}, # 가장 최근 사용자 메시지를 입력으로 전달
                config
            )
            answer = response['answer']

            # 타이핑 효과 구현
            for chunk in answer.split(" "): # 단어 단위로 끊어서 타이핑 효과
                full_response += chunk + " "
                # `message_placeholder`를 업데이트하여 점진적으로 텍스트 표시
                message_placeholder.markdown(f"<span style='font-size:24px;'>{full_response}</span>", unsafe_allow_html=True)
                time.sleep(0.05) # 타이핑 속도 조절 (0.01~0.05 정도가 적당)

            # 최종 답변을 chat_history의 마지막 AI 메시지에 업데이트
            # 이렇게 하면 다음 새로고침 시 이 메시지가 완전한 형태로 표시됩니다.
            chat_history.messages[-1].content = answer

        # 참고 문서 유사도 필터링 및 출력 (유사도 0.4 이상만)
        # 이 부분은 `rag_chain`의 `response`에 포함된 `context`를 직접 사용하는 것이 더 효율적입니다.
        # 현재 코드는 `get_vectorstore([])`를 다시 호출하여 별도로 유사도 검색을 수행하고 있습니다.
        # 이는 중복 작업이며, `rag_chain`이 이미 검색한 문서를 활용하는 것이 좋습니다.
        # 하지만 현재 코드의 로직을 유지하며 스타일만 적용합니다.
        
        embeddings_for_score = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
        vectorstore_for_score = get_vectorstore([]) # 캐시된 단일 벡터스토어 로드 시도
        
        # 마지막 사용자 메시지를 이용하여 다시 유사도 검색 수행
        scored_docs = vectorstore_for_score.similarity_search_with_score(chat_history.messages[-2].content, k=3)

        filtered_docs = []
        for doc, score in scored_docs:
            sim_score = 1 - score / 2 # FAISS (cosine) 기준 변환 (낮은 점수가 높은 유사도)
            if sim_score >= 0.4: # 유사도 0.4 이상만 필터링 (높은 유사도)
                filtered_docs.append(doc)

        # 버튼(Expander) 눌렀을 때만 표시
        if filtered_docs:
            with st.expander("🔎 참고 문서 확인"):
                # 참고 문서 내용을 감싸는 div 추가 (CSS 적용을 위함)
                st.markdown("<div class='reference-docs-content'>", unsafe_allow_html=True)
                for i, doc in enumerate(filtered_docs):
                    source = os.path.basename(doc.metadata.get("source", ""))
                    page = doc.metadata.get("page", None)
                    st.markdown(f"**문서 {i+1}:**") # 문서 번호 표시
                    if source and page is not None:
                        st.markdown(f"- 📄 {source} - {page + 1}쪽")
                    else:
                        st.markdown("- ❔ 출처 없음")
                    st.write(doc.page_content) # 참고 문서 내용 출력
                    st.markdown("---")
                st.markdown("</div>", unsafe_allow_html=True) 
