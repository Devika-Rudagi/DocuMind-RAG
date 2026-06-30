
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
#from google.colab import userdata
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

class DocuMind:
    def __init__(self, pdf_path):
        print("Initializing DocuMind Financial Analyst...")
        self.vectorstore = None
        self.chain = None
        self.pdf_path = pdf_path

    def ingest(self):
        """Ingests the PDF and builds the vector DB"""
        print(f"Loading {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()

        # Splitter - optimized for financial text (slightly larger chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        splits = text_splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")

        # Embeddings (Local & Free)
        print("Indexing...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory="./data/documind_db"
        )
        print("Knowledge Base Ready!")

    def _format_docs(self, docs):
        """Helper to join docs into a single string for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_chain(self):
        #GROQ_API_KEY = userdata.get('LLM-Learn-API-key-02')
        GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

        # 1. Vector Store Setup
        if self.vectorstore is None:
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma(
                persist_directory="./data/documind_db",
                embedding_function=embedding_model
            )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # 2. Contextualization Chain (The "Re-writer") 
        # Solves Pronoun problem
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = (
            contextualize_q_prompt | llm | StrOutputParser() | retriever
        )

        # 3. QA Chain (The "Answerer")
        qa_system_prompt = (
            "You are a Senior Financial Analyst. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you cannot find that information."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = (
            RunnablePassthrough.assign(context=history_aware_retriever)
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # 4. MEMORY UPGRADE: Persistence & Trimming

        # A. Define the Trimmer (Keeps the last ~10 messages to save tokens)
        # It counts tokens and cuts off old messages if they exceed the limit.
        self.trimmer = trim_messages(
            max_tokens=2000,
            strategy="last",
            token_counter=llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        # B. Define the Persistent History Loader
        # This saves every chat to a file 'memory.db' automatically.
        def get_session_history(session_id: str):
            return SQLChatMessageHistory(
                session_id=session_id,
                connection="sqlite:///data/memory.db" #changed from connection_string to connection
            )

        # C. The Final Chain
        # We inject the trimmer between the history fetch and the prompt
        self.conversational_chain = RunnableWithMessageHistory(
            question_answer_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            #output_messages_key="answer"
        )

    def ask(self, query, session_id="investor_01"):
        print(f"\n Analyzing: '{query}'...")

        # 1. Run the chain with session_id
        response = self.conversational_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"Answer: {response}")
        return response

if __name__ == "__main__":
    # 1. Download a 10-K PDF and name it 'apple_10k.pdf'
    pdf_file = "apple_10k.pdf"

    if not os.path.exists(pdf_file):
        print(f"Warning: {pdf_file} not found. Please upload it.")

    app = DocuMind(pdf_file)

    # Check if DB exists to avoid re-ingesting every time (Optimization)
    if os.path.exists("./data/documind_db"):
        print("Database found. Loading...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        app.vectorstore = Chroma(persist_directory="./data/documind_db", embedding_function=embedding_model)
    else:
        app.ingest()

    app.setup_chain()

    # Interactive Loop
    while True:
        q = input("\n[Investor]: ")
        if q.lower() == "exit": break
        app.ask(q)