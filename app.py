from pathlib import Path
import re
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rag.rag_pipeline import build_llm, build_pipeline, rebuild_pipeline_from_uploaded_file
from transformers import T5ForConditionalGeneration, T5Tokenizer
from groqapi import get_response



BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"

app = FastAPI(
    title="LearnIO",
    description="One stop solution for all learning needs",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR))

model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
rag_pipeline = None
rag_llm = None


class DialogueInput(BaseModel):
    dialogue: str


class RAGQueryInput(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class GroqInput(BaseModel):
    message: str


def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = build_pipeline()
    return rag_pipeline


def get_rag_llm():
    global rag_llm
    if rag_llm is None:
        rag_llm = build_llm()
    return rag_llm


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text.strip().lower()


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(targets[0], skip_special_tokens=True)



def answer_with_rag(query: str, top_k: int = 5, score_threshold: float = 0.0) -> dict[str, Any]:
    pipeline = get_rag_pipeline()
    llm = get_rag_llm()
    retrieved_docs = pipeline.retriever.retrieve(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    context = "\n\n".join(
        f"[Document {item['rank']}]\n{item['document']}" for item in retrieved_docs
    )

    if not context:
        context = "No relevant documents were retrieved."

    prompt = (
        "Answer the user's question using only the provided context. "
        "If the context is not enough, say that clearly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke(prompt)
    answer = getattr(response, "content", str(response))
    return {"answer": answer, "sources": retrieved_docs}


def set_rag_pipeline_from_upload(file: UploadFile):
    global rag_pipeline
    rag_pipeline = rebuild_pipeline_from_uploaded_file(file)
    return rag_pipeline


@app.post("/summarize")
async def summarize(dialogue_entered: DialogueInput):
    summary = summarize_dialogue(dialogue_entered.dialogue)
    return {"summary": summary}


@app.post("/rag/retrieve")
async def rag_retrieve(query_input: RAGQueryInput):
    try:
        pipeline = get_rag_pipeline()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrieved_docs = pipeline.retriever.retrieve(
        query=query_input.query,
        top_k=query_input.top_k,
        score_threshold=query_input.score_threshold,
    )
    return {"results": retrieved_docs}


@app.post("/rag/ask")
async def rag_ask(query_input: RAGQueryInput):
    try:
        return answer_with_rag(
            query=query_input.query,
            top_k=query_input.top_k,
            score_threshold=query_input.score_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/groq/chat")
async def groq_chat(groq_input: GroqInput):
    try:
        response = get_response(groq_input.message)
        return {"response": response}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose a file to upload.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    try:
        pipeline = set_rag_pipeline_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunk_count = pipeline.vector_store.collection.count()
    return {
        "message": f"Indexed {file.filename} successfully.",
        "filename": file.filename,
        "chunks_indexed": chunk_count,
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/summarizer-ui", response_class=HTMLResponse)
async def summarizer_ui(request: Request):
    return templates.TemplateResponse(request=request, name="summarizer.html")


@app.get("/rag-ui", response_class=HTMLResponse)
async def rag_ui(request: Request):
    return templates.TemplateResponse(request=request, name="rag.html")


@app.get("/groq-chat-ui", response_class=HTMLResponse)
async def groq_chat_ui(request: Request):
    return templates.TemplateResponse(request=request, name="groq-chat.html")
