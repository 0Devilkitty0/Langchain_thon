import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader # DirectoryLoader를 임포트합니다.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# OpenAI API Key 설정
# Streamlit의 secrets에 'OPENAI_API_KEY'를 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ====================================
# PRE-PROCESSING 단계
# ====================================

class DocumentProcessor:
    """문서 전처리를 담당하는 클래스"""

    @staticmethod
    @st.cache_resource
    def load_documents(directory_path: str):
        """
        1. 문서 로드 (Document Load)
        - 지정된 디렉토리에서 지원하는 모든 형식의 파일(.pdf, .txt, .docx 등)을 읽어들입니다.
        """
        if not os.path.isdir(directory_path):
            st.error(f"오류: '{directory_path}' 디렉토리를 찾을 수 없습니다.")
            st.stop()

        # DirectoryLoader를 사용하여 다양한 파일 형식 로드
        # show_progress=True로 설정하여 로딩 진행 상황을 터미널에 표시합니다.
        # use_multithreading=True로 설정하여 여러 파일을 동시에 빠르게 로드합니다.
        loader = DirectoryLoader(directory_path, glob="**/*.*", show_progress=True, use_multithreading=True)
        
        st.info(f"📁 '{directory_path}' 디렉토리에서 모든 문서를 로드합니다...")
        
        try:
            documents = loader.load()
        except Exception as e:
            st.error(f"문서 로드 중 오류가 발생했습니다: {e}")
            documents = [] # 오류 발생 시 빈 리스트로 초기화

        if not documents:
            st.error("로드할 문서가 없습니다. 'data' 디렉토리를 확인해주세요.")
            st.stop()

        st.info(f"📄 총 문서 로드 완료: {len(documents)} 개의 문서")
        return documents

    @staticmethod
    def split_text(documents, chunk_size=1000, chunk_overlap=200):
        """
        2. Text Split (청크 분할)
        - 불러온 문서를 chunk 단위로 분할합니다.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        st.info(f"✂️ 텍스트 분할 완료: {len(split_docs)} 청크")
        return split_docs

    @staticmethod
    @st.cache_resource
    def create_vector_store(_split_docs):
        """
        4. DB 저장 (Vector Store)
        - 변환된 벡터를 FAISS DB에 저장합니다.
        """
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        vectorstore = FAISS.from_documents(_split_docs, embeddings)
        st.success("💾 벡터 DB 저장 완료!")
        return vectorstore

# ====================================
# RUNTIME 단계
# ====================================

class RAGRetriever:
    """검색기(Retriever) 관리 클래스"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_retriever(self, search_type="similarity", k=5):
        """
        1. 검색 (Retrieve)
        - Vector DB에서 관련 문서를 찾는 검색기를 생성합니다. (k값을 5로 조정하여 더 많은 문맥 참조)
        """
        retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return retriever

class PromptManager:
    """프롬프트 관리 클래스"""

    @staticmethod
    def get_contextualize_prompt():
        """
        2. 프롬프트 (Prompt) - 대화 맥락화
        - 채팅 기록을 바탕으로 후속 질문을 독립적인 질문으로 재구성합니다.
        """
        contextualize_q_system_prompt = """주어진 채팅 히스토리와 최신 사용자 질문을 바탕으로,
        채팅 히스토리 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요.
        질문에 답하지 말고, 필요시 재구성하거나 그대로 반환하세요."""

        return ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

    @staticmethod
    def get_qa_prompt():
        """
        2. 프롬프트 (Prompt) - 질문 답변
        - 검색된 문맥을 바탕으로 답변을 생성하기 위한 프롬프트입니다.
        """
        # "헌법 전문가"에서 "AI 어시스턴트"로 좀 더 일반적인 역할로 변경
        qa_system_prompt = """당신은 주어진 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
        제공된 검색 결과를 바탕으로 질문에 답변하세요.

        답변 규칙:
        - 정확한 정보만 제공하세요.
        - 모르는 내용에 대해서는 '문서에 관련 정보가 없습니다.'라고 솔직하게 답변하세요.
        - 한국어로 친절하고 상세하게 답변하세요.
        - 가능하다면 관련된 문서의 출처(source)와 페이지(page)를 함께 언급해주세요.

        검색된 문맥:
        {context}"""

        return ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

class LLMManager:
    """
    3. 언어 모델 (LLM) 관리
    - GPT-4o-mini 등 다양한 모델을 선택할 수 있습니다.
    """

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
        )

    def get_llm(self):
        return self.llm

class RAGChain:
    """RAG 체인 구성 및 관리"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt_manager = PromptManager()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, prompt_manager.get_contextualize_prompt()
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt_manager.get_qa_prompt())
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def get_conversational_chain(self, chat_history):
        return RunnableWithMessageHistory(
            self.chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

# ====================================
# 메인 애플리케이션
# ====================================

@st.cache_resource
def initialize_rag_system(model_name):
    """RAG 시스템 초기화"""
    st.info("🔄 RAG 시스템 초기화 중...")
    documents = DocumentProcessor.load_documents("data")
    split_docs = DocumentProcessor.split_text(documents)
    vectorstore = DocumentProcessor.create_vector_store(split_docs)
    rag_retriever = RAGRetriever(vectorstore)
    retriever = rag_retriever.get_retriever()
    llm_manager = LLMManager(model_name)
    llm = llm_manager.get_llm()
    rag_chain = RAGChain(retriever, llm)
    st.success("✅ RAG 시스템 초기화 완료!")
    return rag_chain

def format_output(response):
    """결과 포맷팅"""
    answer = response.get('answer', '답변을 생성할 수 없습니다.')
    context = response.get('context', [])
    return {
        'answer': answer,
        'context': context,
        'source_count': len(context)
    }

# ====================================
# Streamlit UI
# ====================================

def main():
    # 페이지 제목과 아이콘을 좀 더 일반적인 내용으로 변경
    st.set_page_config(
        page_title="RAG 문서 Q&A 챗봇",
        page_icon="🤖",
        layout="wide"
    )

    st.header("🤖 RAG 기반 문서 Q&A 챗봇 💬")
    st.markdown("`data` 폴더의 문서(PDF, TXT, DOCX 등)를 기반으로 질문에 답변합니다.")

    with st.sidebar:
        st.header("🔧 설정")
        model_option = st.selectbox(
            "GPT 모델 선택",
            ("gpt-4o-mini", "gpt-3.5-turbo-0125", "gpt-4o"),
            help="사용할 GPT 모델을 선택하세요"
        )

    rag_chain = initialize_rag_system(model_option)
    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    conversational_rag_chain = rag_chain.get_conversational_chain(chat_history)

    # 초기 메시지를 일반적인 내용으로 변경
    if not chat_history.messages:
        chat_history.add_ai_message("안녕하세요! `data` 폴더의 문서에 대해 무엇이든 물어보세요! 📚")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("문서에 대해 궁금한 점을 입력해주세요..."):
        st.chat_message("human").write(prompt)

        with st.chat_message("ai"):
            with st.spinner("🧐 문서를 검색하고 답변을 생성 중입니다..."):
                try:
                    config = {"configurable": {"session_id": "rag_chat"}}
                    response = conversational_rag_chain.invoke({"input": prompt}, config)
                    
                    formatted_response = format_output(response)
                    st.write(formatted_response['answer'])

                    with st.expander(f"📄 참고 문서 ({formatted_response['source_count']}개)"):
                        if formatted_response['context']:
                            for i, doc in enumerate(formatted_response['context']):
                                st.markdown(f"**📖 문서 {i+1}**")
                                source = doc.metadata.get('source', '출처 정보 없음')
                                st.markdown(f"**출처:** `{source}`")
                                
                                # 페이지 번호는 PDF 파일에만 존재할 수 있음
                                if 'page' in doc.metadata:
                                    page = doc.metadata.get('page')
                                    st.markdown(f"**페이지:** {page + 1}")

                                st.text_area(
                                    f"문서 {i+1} 내용",
                                    doc.page_content,
                                    height=150,
                                    key=f"doc_{i}",
                                    label_visibility="collapsed"
                                )
                                if i < len(formatted_response['context']) - 1:
                                    st.markdown("---")
                        else:
                            st.info("답변에 참고한 문서를 찾을 수 없습니다.")
                
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    st.info("다시 시도해주세요.")

if __name__ == "__main__":
    main()
