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

#오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("MY_OPENAI_API_KEY")

#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# FAISS 벡터스토어 생성
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 로컬에 저장 (옵션: 나중에 get_vectorstore에서 사용 가능)
    vectorstore.save_local("faiss_index")
    return vectorstore

# FAISS 벡터스토어를 로컬에서 로드하거나, 없으면 새로 생성
@st.cache_resource
def get_vectorstore(_docs):
    if os.path.exists("faiss_index/index.faiss") and os.path.exists("faiss_index/index.pkl"):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store(_docs)

def refine_question(user_input: str) -> str:
    mapping = {
        "아프면": "판단능력이 없을 때",
        "내 재산은 누가": "재산 관리는 누가",
        "누가 나를 돌봐줄 수": "후견인이 어떤 역할을 할 수",
        "내가 아프면": "내가 판단능력이 없게 되면",
    }

    for key, val in mapping.items():
        if key in user_input:
            user_input = user_input.replace(key, val)

    return user_input


# data 폴더의 PDF 전부 로드해서 벡터 DB 만들고, 검색 + 히스토리까지 포함한 전체 Chain 구성
@st.cache_resource
def initialize_components(selected_model):
    data_dir = "./data"
    all_pages = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            try:
                pages = load_and_split_pdf(file_path)
                all_pages.extend(pages)
            except Exception as e:
                st.warning(f"⚠️ {filename} 불러오기 실패: {e}")

    if not all_pages:
        st.error("❌ data 폴더에 유효한 PDF 문서가 없습니다.")
        return None

    # 벡터스토어 생성 및 retriever 추출
    vectorstore = get_vectorstore(all_pages)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # 히스토리 기반 retriever를 위한 system prompt
    contextualize_q_system_prompt = """주어진 채팅 히스토리와 최신 사용자 질문을 바탕으로,
    채팅 히스토리 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요.

    또한, 일상적인 표현이거나 모호한 표현이 있을 경우,
    문서에서 자주 쓰이는 공식 용어나 법률 용어(예: 후견인, 재산관리, 의사결정능력, 치매 등)로 바꾸어 표현하세요.

    질문에 답하지 말고, 필요시 재구성하거나 그대로 반환하세요.
    """


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    qa_system_prompt = """당신은 주어진 문서를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

    당신의 주요 임무는 치매를 준비하는 노인을 위한 국가 지원 제도, 복지 혜택, 의료 및 돌봄 정보 등 치매 전반에 대한 질문에 대해 정확하고 유익한 정보를 제공하는 것입니다.  
    답변은 간결하고 **최대한 8줄 이내로 설명**해주고, 어린이도 이해할 수 있을 정도로 쉬운 언어로 설명해 주세요.

    ---  
    **답변 기준 및 규칙**  

    1. 아래에 제공된 문서(context)가 존재할 경우
        - 반드시 **context 내용만을 기반으로** 답변하세요.  
        - 일반적인 배경지식은 사용하지 마세요.
        - 출처나 쪽수는 **표시하지 마세요.**
        - 연락처를 물어보는경우 연락처를 안내해주세요.  
        

    2. 아래 context가 비어 있을 경우
        - GPT 모델이 알고 있는 일반적인 지식만을 사용해 답변하세요.  
        - 이 경우, 반드시 다음 문장을 **답변의 첫머리에 줄바꿈 2번 후 출력**하세요 : 이 답변은 제가 가진 일반적인 정보로 알려 드리는 거예요.

    3. 문서에서 제공되지 않는 정보의 경우
        - **답변의 첫머리에 줄바꿈 2번 후 출력**하세요 : 이 답변은 제가 가진 일반적인 정보로 알려 드리는 거예요.

         ---  
         참고 문서 (context):  
         {context}

     """


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # LLM, 히스토리 포함 retriever, QA chain 구성
    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 최종 RAG chain 반환
    return rag_chain


# Streamlit UI
st.set_page_config(page_title="메모리네비 💬📚", page_icon="🧭", layout="centered")

st.markdown("""
<div class="title-section">
    <h1>🧭 메모리네비</h1>
    <p>어르신을 위한 치매 관련 정보 도우미입니다.<br>
    궁금한 내용을 편하게 물어보세요!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 전체 채팅 메시지 기본 글씨 크기 */
.stChatMessage {
    font-size: 48px;
    line-height: 1.6;
}

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

/* AI 답변 텍스트 크기 (기본) */
.stChatMessage p {
    font-size: 24px !important;    
    line-height: 1.6 !important;
}

/* 첫 번째 AI 버블 여백 및 높이 */
div[data-testid="stChatMessage"][data-variant="assistant"]:first-of-type {
    padding: 1.2rem 1.5rem !important;
    min-height: 96px !important;
}

/* 입력창 자체 높이 및 정렬 */
[data-testid="stChatInput"] > div:first-child {
    min-height: 64px !important;
    display: flex;
    align-items: center;
    padding: 0 1rem !important;
}

/* 입력창 텍스트 영역 */
[data-testid="stChatInput"] textarea {
    font-size: 20px !important;
    line-height: 1.6 !important;
    padding: 0.6rem 0.5rem !important;
}

/* 입력창 placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 20px !important;
    opacity: 0.7;
}

/* AI 메시지 전체 폭 넓히기 */
div[data-testid="stChatMessage"][data-variant="assistant"] {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* <b>, <strong>, <li> 등 강조 텍스트도 글자 크게 */
.stMarkdown b,
.stMarkdown strong,
.stMarkdown li,
.stMarkdown p {
    font-size: 24px !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)




# chat_history = StreamlitChatMessageHistory(key="chat_messages")


# option = st.selectbox("사용하실 GPT 모델을 선택해주세요. (숫자가 높을수록 좋은 모델입니다)", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
selected_model = "gpt-4o-mini"
rag_chain = initialize_components(selected_model)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 세션 상태 초기화 (문맥/채팅 메시지)
if "context" not in st.session_state:
    st.session_state["context"] = []

# 챗 히스토리에 메시지가 없으면, 초기 환영 메시지 추가
# if not chat_history.messages:
#     chat_history.add_ai_message("치매에 대해 무엇이든 물어보세요!")

# 이전 대화 메시지 출력
for msg in chat_history.messages:
    # 챗 메시지 출력 시 크기 조절
    if msg.type == "human":
        # 사용자의 메시지는 파란색으로, 폰트 크기 24px
        st.chat_message("human").markdown(f"<span style='font-size:24px; color:#007BFF;'>{msg.content}</span>", unsafe_allow_html=True)
    else: # msg.type == "ai"
        # AI의 메시지는 기본 색상으로, 폰트 크기 24px
        st.chat_message("ai").markdown(f"<span style='font-size:24px;'>{msg.content}</span>", unsafe_allow_html=True)

    
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 사용자가 질문을 입력하는 부분
if prompt_message := st.chat_input("치매에 대해 궁금한 점을 여기에 입력해 주세요."):
    refined_message = refine_question(prompt_message)  # 질문 리포맷
    if conversational_rag_chain: # 챗봇이 활성화된 경우에만 질문 처리
        # 사용자의 메시지 출력 (글씨 크기 및 색상 조절)
        st.chat_message("human").markdown(f"<span style='font-size:24px; color:#007BFF;'>{prompt_message}</span>", unsafe_allow_html=True)

        with st.chat_message("ai"):
            with st.spinner("생각 중입니다... 🧐"):
                config = {"configurable": {"session_id": "any"}}
                response = conversational_rag_chain.invoke(
                    {"input": prompt_message},
                    config
                )

                answer = response['answer']
                # AI의 답변 출력 (글씨 크기 조절)
                placeholder = st.empty()
                for i in range(len(answer)):
                    placeholder.markdown(f"<span style='font-size:24px;'>{answer[:i+1]}</span>", unsafe_allow_html=True)
                    time.sleep(0.01)  # 타이핑 속도 조절 (빠르게 하고 싶으면 줄이기)

            # 참고 문서 유사도 필터링 및 출력 (유사도 0.3 이상만)
            vectorstore = get_vectorstore([])  # 기존 vectorstore 다시 불러오기
            scored_docs = vectorstore.similarity_search_with_score(prompt_message, k=2)

            filtered_docs = []
            for doc, score in scored_docs:
                sim_score = 1 - score / 2  # FAISS (cosine) 기준 변환
                if sim_score >= 0.4:
                    filtered_docs.append(doc)  # 점수는 안 씀

            # 버튼(Expander) 눌렀을 때만 표시
            if filtered_docs:
                with st.expander("🔎 참고 문서 확인"):
                    for doc in filtered_docs:
                        source = os.path.basename(doc.metadata.get("source", ""))
                        page = doc.metadata.get("page", None)
                        if source and page is not None:
                            st.markdown(f"- 📄 {source} - {page + 1}쪽")
                        else:
                            st.markdown("- ❔ 출처 없음")

# 초기 환영 메시지 (AI 버블로 넣기)
if not chat_history.messages:
    st.markdown("""
    <div style='text-align: center; margin-top: 1.5rem;'>
        <h2 style='margin: 0; font-size: 28px;'>치매에 대해 무엇이든 물어보세요!</h2>
    </div>
    """, unsafe_allow_html=True)
