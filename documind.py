import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from google.colab import userdata

# --- CONFIGURATION ---

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
            persist_directory="./documind_db"
        )
        print("Knowledge Base Ready!")

    def _format_docs(self, docs):
        """Helper to join docs into a single string for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def setup_chain(self):
        GROQ_API_KEY = userdata.get('LLM-Learn-API-key-02')
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

        if self.vectorstore is None:
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma(
                persist_directory="./documind_db",
                embedding_function=embedding_model
            )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # Professional Financial Prompt
        system_prompt = (
            "You are a Senior Financial Analyst. "
            "Use the retrieved 10-K report context to answer the investor's question. "
            "If the answer is not in the context, strictly say 'I cannot find that information in the report'. "
            "Keep answers factual, concise, and professional."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Step 1: Retrieve documents AND keep the input question
        # This creates a dictionary: {'context': [Doc1, Doc2], 'input': 'User Query'}
        entry_point = RunnableParallel(
            context=retriever,
            input=RunnablePassthrough()
        )

        # Step 2: Generate the Answer
        # We take the dictionary from Step 1, format the docs for the LLM, and get a string response
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: self._format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 3: Combine them
        # 'entry_point' gets the docs. '.assign(answer=...)' adds the LLM result to that dictionary.
        # Final Result: {'context': [Doc objects], 'input': 'Query', 'answer': 'The revenue is...'}
        self.chain = entry_point.assign(answer=rag_chain_from_docs)

    def ask(self, query):
        print(f"\n Analyzing: '{query}'...")
        start_time = time.time()

        response = self.chain.invoke(query)

        elapsed = time.time() - start_time
        print(f"Answer ({elapsed:.2f}s):")
        print(response)


        print("\nSources:")
        unique_pages = set()
        for doc in response['context']:
            page = doc.metadata.get('page', 'Unknown')
            unique_pages.add(page)
            # Preview the text used
            print(f"- Page {page}: {doc.page_content[:60]}...")

        return response

# --- RUN THE APP ---
if __name__ == "__main__":
    pdf_file = "apple_10k.pdf"

    app = DocuMind(pdf_file)

    # Check if DB exists to avoid re-ingesting every time (Optimization)
    if os.path.exists("./documind_db"):
        print("Database found. Loading...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        app.vectorstore = Chroma(persist_directory="./documind_db", embedding_function=embedding_model)
    else:
        app.ingest()

    app.setup_chain()

    # Interactive Loop
    while True:
        q = input("\n[Investor]: ")
        if q.lower() == "exit": break
        app.ask(q)