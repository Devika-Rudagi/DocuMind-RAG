from dotenv import load_dotenv
load_dotenv()

import os
import boto3
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from documind_v2 import DocuMind

# S3 setup
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)
BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

def upload_to_s3(local_path: str, filename: str) -> str:
    s3_key = f"uploads/{filename}"
    s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
    return s3_key

def download_from_s3(s3_key: str, local_path: str):
    s3_client.download_file(BUCKET_NAME, s3_key, local_path)

# DocuMind agent
agent = DocuMind(pdf_path="placeholder.pdf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists("./data/documind_db"):
        print("Existing knowledge base found — loading...")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        agent.vectorstore = Chroma(
            persist_directory="./data/documind_db",
            embedding_function=embedding_model
        )
        agent.setup_chain()
        print("Knowledge base loaded. Ready to answer questions.")
    yield

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        s3_key = upload_to_s3(tmp_path, file.filename)
        print(f"PDF uploaded to S3: {s3_key}")
        agent.pdf_path = tmp_path
        agent.ingest()
        agent.setup_chain()
    finally:
        os.unlink(tmp_path)

    return {
        "message": f"{file.filename} ingested successfully",
        "s3_key": s3_key
    }

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