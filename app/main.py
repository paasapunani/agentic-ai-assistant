from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import run_agent
from .config import settings

app = FastAPI(title="Agentic AI Assistant", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    plan: str
    steps: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.provider}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        out = run_agent(req.message, req.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(thread_id=req.thread_id, **out)
