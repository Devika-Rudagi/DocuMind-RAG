from dotenv import load_dotenv
load_dotenv()

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from documind_v2 import DocuMind

agent = DocuMind(pdf_path="apple_10k.pdf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE when server starts
    if os.path.exists("./data/documind_db"):
        print("Existing knowledge base found — loading...")
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        agent.vectorstore = Chroma(
            persist_directory="./data/documind_db",
            embedding_function=embedding_model
        )
        agent.setup_chain()
        print("Knowledge base loaded. Ready to answer questions.")
    yield
    # Runs on shutdown (cleanup if needed — nothing for now)

app = FastAPI(title="DocuMind API", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = f"./data/{file.filename}"
    os.makedirs("./data", exist_ok=True)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    agent.pdf_path = save_path
    agent.ingest()
    agent.setup_chain()

    return {"message": f"{file.filename} ingested successfully"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not os.path.exists("./data/documind_db"):
        raise HTTPException(
            status_code=400,
            detail="No document has been ingested yet. Call /upload first."
        )

    try:
        answer = agent.ask(request.question, session_id=request.session_id)
    except AttributeError:
        raise HTTPException(
            status_code=400,
            detail="DocuMind chain not initialized. Call /upload first."
        )

    return QueryResponse(answer=answer)