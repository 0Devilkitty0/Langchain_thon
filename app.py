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
openai_api_key = os.getenv("MY_OPENAI_API_KEY")
if openai_api_key is None:
    st.error("오류: OpenAI API 키가 설정되지 않았습니다. 'MY_OPENAI_API_KEY' 환경 변수를 설정해주세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# 캐시 리소스로 PDF 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 캐시 리소스로 FAISS 벡터스토어 생성
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# 캐시 리소스로 벡터스토어 로드 또는 생성
@st.cache_resource
def get_vectorstore(_docs):
    faiss_index_path = "faiss_index"
    if os.path.exists(os.path.join(faiss_index_path, "index.faiss")) and \
       os.path.exists(os.path.join(faiss_index_path, "index.pkl")):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
        return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store(_docs)

# 캐시 리소스로 LangChain 컴포넌트 초기화
@st.cache_resource
def initialize_components(selected_model):
    data_dir = "./data"
    all_pages = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        st.info(f"'{data_dir}' 디렉토리가 생성되었습니다. PDF 파일을 이 안에 넣어주세요.")
        return None 

    pdf_found = False
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            try:
                pages = load_and_split_pdf(file_path)
                all_pages.extend(pages)
                pdf_found = True
            except Exception as e:
                st.warning(f"⚠️ {filename} 불러오기 실패: {e}")

    if not pdf_found:
        st.error("❌ data 폴더에 유효한 PDF 문서가 없습니다.")
        return None

    vectorstore = get_vectorstore(all_pages)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """주어진 채팅 히스토리와 최신 사용자 질문을 바탕으로,
        채팅 히스토리 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요.
        질문에 답하지 말고, 필요시 재구성하거나 그대로 반환하세요."""
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
    **답변 기준 및 규칙** 1. 아래에 제공된 문서(context)가 존재할 경우
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

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain

# Streamlit UI 타이틀 및 CSS
st.markdown("""
<div class="title-section">
    <h1>🧭 메모리네비</h1>
    <p>어르신을 위한 치매 관련 정보 도우미입니다.<br>
    궁금한 내용을 편하게 물어보세요!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ... (기존 CSS 스타일 유지) ... */
.stTextInput > div > input {
    font-size: 24px !important;
}
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
div[data-testid="stChatMessage"][data-variant="assistant"] {
    font-size: 24px !important;
    line-height: 1.8 !important;
}
div[data-testid="stChatMessage"][data-variant="assistant"] p {
    font-size: 24px !important;
    line-height: 1.8 !important;
}
div[data-testid="stChatMessage"][data-variant="user"] {
    font-size: 24px !important;
    line-height: 1.6 !important;
}
div[data-testid="stChatMessage"][data-variant="user"] p {
    font-size: 24px !important;
    line-height: 1.6 !important;
}
.reference-docs-content,
.reference-docs-content p,
.reference-docs-content span,
.reference-docs-content li {
    font-size: 18px !important;
    line-height: 1.5 !important;
}
div[data-testid="stChatMessage"][data-variant="assistant"]:first-of-type {
    padding: 1.2rem 1.5rem !important;
    min-height: 96px !important;
}
[data-testid="stChatInput"] > div:first-child {
    min-height: 64px !important;
    display: flex;
    align-items: center;
    padding: 0 1rem !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 20px !important;
    line-height: 1.6 !important;
    padding: 0.6rem 0.5rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 20px !important;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)


selected_model = "gpt-4o-mini"
rag_chain = initialize_components(selected_model)

# StreamlitChatMessageHistory를 세션 상태에 저장하고 관리
# 이 객체가 st.session_state["chat_messages"]와 연결됩니다.
if "chat_history_obj" not in st.session_state:
    st.session_state.chat_history_obj = StreamlitChatMessageHistory(key="chat_messages")

# 초기 환영 메시지 (StreamlitChatMessageHistory에 없으면 추가)
if not st.session_state.chat_history_obj.messages:
    st.session_state.chat_history_obj.add_ai_message("치매에 대해 무엇이든 물어보세요! 🧠✨")

#    `st.chat_message`는 메시지를 표시하는 역할만 하도록 합니다.
for msg in st.session_state.chat_history_obj.messages:
    if msg.type == "human":
        with st.chat_message("human"):
            st.markdown(f"<span style='font-size:24px; color:#007BFF;'>{msg.content}</span>", unsafe_allow_html=True)
    elif msg.type == "ai":
        if msg.content != "": # 내용이 있는 AI 메시지만 그립니다.
            with st.chat_message("ai"):
                st.markdown(f"<span style='font-size:24px;'>{msg.content}</span>", unsafe_allow_html=True)


# rag_chain이 성공적으로 초기화되었을 때만 conversational_rag_chain을 생성
if rag_chain:
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: st.session_state.chat_history_obj, # 세션에 저장된 StreamlitChatMessageHistory 객체 사용
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
else:
    st.info("PDF 문서가 없거나 로드에 실패하여 챗봇을 사용할 수 없습니다. 'data' 폴더에 PDF 파일을 넣어주세요.")
    conversational_rag_chain = None


# 사용자가 질문을 입력하는 부분
if prompt_message := st.chat_input("치매에 대해 궁금한 점을 여기에 입력해 주세요."):
    if conversational_rag_chain:
        # 사용자 메시지를 chat_history_obj에 추가
        st.session_state.chat_history_obj.add_user_message(prompt_message)
        
        with st.chat_message("ai"):
            message_placeholder = st.empty() # 이 placeholder에 응답을 점진적으로 표시

            full_response = ""
            with st.spinner("생각 중입니다... 🧐"):
                config = {"configurable": {"session_id": "any"}}
                
                response = conversational_rag_chain.invoke(
                    {"input": prompt_message}, 
                    config
                )
                answer = response['answer']

                # 타이핑 효과
                for chunk in answer.split(" "):
                    full_response += chunk + " "
                    message_placeholder.markdown(f"<span style='font-size:24px;'>{full_response}</span>", unsafe_allow_html=True)
                    time.sleep(0.05)

# chat_history 대신 일반 리스트 사용
if "messages" not in st.session_state:
    st.session_state.messages = []

# 초기 환영 메시지 (메시지 리스트에 없으면 추가)
if not st.session_state.messages:
    st.session_state.messages.append(AIMessage(content="치매에 대해 무엇이든 물어보세요! 🧠✨"))

# 이전 대화 메시지 출력 (st.session_state.messages에서 읽어서 표시)
for msg in st.session_state.messages:
    if msg.type == "human":
        with st.chat_message("human"):
            st.markdown(f"<span style='font-size:24px; color:#007BFF;'>{msg.content}</span>", unsafe_allow_html=True)
    else: # msg.type == "ai"
        # AI 응답이 비어있지 않은 경우에만 표시 (타이핑 효과 중 빈 버블 중복 방지)
        if msg.content != "":
            with st.chat_message("ai"):
                st.markdown(f"<span style='font-size:24px;'>{msg.content}</span>", unsafe_allow_html=True)

# rag_chain이 성공적으로 초기화되었을 때만 conversational_rag_chain을 생성
if rag_chain:
    # StreamlitChatMessageHistory 인스턴스를 세션 상태에 유지
    if "chat_history_obj" not in st.session_state:
        st.session_state.chat_history_obj = StreamlitChatMessageHistory(key="chat_messages")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: st.session_state.chat_history_obj, # 세션에 저장된 객체 사용
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )
else:
    st.info("PDF 문서가 없거나 로드에 실패하여 챗봇을 사용할 수 없습니다. 'data' 폴더에 PDF 파일을 넣어주세요.")
    conversational_rag_chain = None

if not st.session_state.chat_history_obj.messages:
    st.session_state.chat_history_obj.add_ai_message("치매에 대해 무엇이든 물어보세요! 🧠✨")

for msg in st.session_state.chat_history_obj.messages:
  
    if msg.type == "human":
        with st.chat_message("human"):
            st.markdown(f"<span style='font-size:24px; color:#007BFF;'>{msg.content}</span>", unsafe_allow_html=True)
    elif msg.type == "ai" and msg.content != "": # 내용이 있는 AI 메시지만 그립니다.
        with st.chat_message("ai"):
            st.markdown(f"<span style='font-size:24px;'>{msg.content}</span>", unsafe_allow_html=True)


# 사용자가 질문을 입력하는 부분
if prompt_message := st.chat_input("치매에 대해 궁금한 점을 여기에 입력해 주세요."):
    if conversational_rag_chain:
        # 1. 사용자 메시지를 chat_history_obj에 추가
        st.session_state.chat_history_obj.add_user_message(prompt_message)
        
        st.session_state.chat_history_obj.add_ai_message("") 
        
        # 3. 앱을 새로고침하여 사용자 메시지(새로 추가된 것)가 표시되도록 합니다.
        #    이때 빈 AI 버블은 위 렌더링 루프에서 건너뛰어져서 보이지 않습니다.
        st.rerun() 

if st.session_state.chat_history_obj.messages and \
   st.session_state.chat_history_obj.messages[-1].type == "ai" and \
   st.session_state.chat_history_obj.messages[-1].content == "":
    
    with st.chat_message("ai"):
        message_placeholder = st.empty() 
        full_response = ""

        with st.spinner("생각 중입니다... 🧐"):
            config = {"configurable": {"session_id": "any"}}
            
            # conversational_rag_chain.invoke 호출 시, LangChain이 내부적으로
            # st.session_state.chat_history_obj를 사용하여 전체 대화 히스토리를 관리합니다.
            response = conversational_rag_chain.invoke(
                {"input": st.session_state.chat_history_obj.messages[-2].content}, # 가장 최근 사용자 메시지
                config
            )
            answer = response['answer']

            # 타이핑 효과
            for chunk in answer.split(" "):
                full_response += chunk + " "
                message_placeholder.markdown(f"<span style='font-size:24px;'>{full_response}</span>", unsafe_allow_html=True)
                time.sleep(0.05)
                
            st.session_state.chat_history_obj.messages[-1].content = answer

        # 참고 문서 유사도 필터링 및 출력
        embeddings_for_score = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
        vectorstore_for_score = get_vectorstore([])
        
        # 가장 최근 사용자 메시지로 유사도 검색
        scored_docs = vectorstore_for_score.similarity_search_with_score(st.session_state.chat_history_obj.messages[-2].content, k=3)

        filtered_docs = []
        for doc, score in scored_docs:
            sim_score = 1 - score / 2 # 코사인 유사도 점수 변환
            if sim_score >= 0.4:
                filtered_docs.append(doc)

        if filtered_docs:
            with st.expander("🔎 참고 문서 확인"):
                st.markdown("<div class='reference-docs-content'>", unsafe_allow_html=True)
                for i, doc in enumerate(filtered_docs):
                    source = os.path.basename(doc.metadata.get("source", ""))
                    page = doc.metadata.get("page", None)
                    st.markdown(f"**문서 {i+1}:**")
                    if source and page is not None:
                        st.markdown(f"- 📄 {source} - {page + 1}쪽")
                    else:
                        st.markdown("- ❔ 출처 없음")
                    st.write(doc.page_content)
                    st.markdown("---")
                st.markdown("</div>", unsafe_allow_html=True)
