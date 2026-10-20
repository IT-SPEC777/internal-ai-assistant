import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from .auth import current_user
from .llm import LLMError
from .rag import answer_query
from .retrieval import get_retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_retrieval)
    yield


app = FastAPI(title="Internal AI Assistant", version="0.2.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    refused: bool
    retrieved_count: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/whoami")
async def whoami(user: dict = Depends(current_user)):
    return {"id": user["id"], "name": user["name"], "groups": user["groups"]}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(current_user)):
    if not req.message.strip():
        raise HTTPException(400, "Пустое сообщение")
    try:
        result = await answer_query(req.message, user)
    except LLMError as e:
        raise HTTPException(503, str(e))
    return ChatResponse(**result)