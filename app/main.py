from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.rag import NCDOTChatbot

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
state: dict[str, NCDOTChatbot] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not Path(config.CHROMA_DIR).exists():
        print("WARNING: vector store not found. Run `python -m app.ingest` first.")
    state["bot"] = NCDOTChatbot()
    yield


app = FastAPI(title="NCDOT Virtual Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)


class Source(BaseModel):
    id: int
    source: str | None = None
    page: int | None = None
    category: str | None = None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[Source]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return await run_in_threadpool(
            state["bot"].ask, req.message.strip(), req.session_id
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}") from exc


@app.post("/reset/{session_id}")
async def reset(session_id: str):
    state["bot"].reset(session_id)
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/config.js")
async def frontend_config():
    return FileResponse(STATIC_DIR / "config.js", media_type="application/javascript")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
