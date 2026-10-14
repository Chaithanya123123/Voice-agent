"""FastAPI backend: PDF upload + chunk viewer + text chat + voice hooks.

Run:
    .\\.venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings
from ingest import (
    delete_document,
    ingest_single_file,
    list_chunks,
    list_documents,
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Singletons: loaded once, reused by /chat and /chunks (fixes 3x reload).
    embedder = SentenceTransformer(settings.embedding_model)
    client = QdrantClient(path=settings.qdrant_path)
    state["embedder"] = embedder
    state["client"] = client
    yield
    try:
        client.close()
    except Exception:
        pass
    state.clear()


app = FastAPI(title="Nano Voice Agents API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    style: Literal["voice", "text"] = "text"


class ChatSource(BaseModel):
    source: str | None = None
    headings: list[str] = []
    page_no: int | None = None
    score: float | None = None
    start: str = ""


class ChatResponse(BaseModel):
    answer: str
    context: str
    sources: list[ChatSource]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    ok = "embedder" in state and "client" in state
    docs = 0
    try:
        if ok:
            info = state["client"].get_collection(settings.collection_name)
            docs = info.points_count or 0
    except Exception:
        pass
    return {"ok": ok, "points": docs, "collection": settings.collection_name}


@app.get("/documents")
def documents():
    try:
        return {"documents": list_documents(state["client"])}
    except Exception as e:
        raise HTTPException(500, f"Qdrant not ready (run ingest first): {e}")


@app.get("/chunks")
def chunks(source: str | None = None, limit: int = Query(100, ge=1, le=500)):
    try:
        return {"chunks": list_chunks(state["client"], source=source, limit=limit)}
    except Exception as e:
        raise HTTPException(500, f"Could not read chunks: {e}")


def _ingest_saved_file(saved: Path):
    client: QdrantClient = state["client"]
    embedder: SentenceTransformer = state["embedder"]
    ingest_single_file(saved, client=client, embedder=embedder)


@app.post("/documents")
async def upload_document(file: UploadFile = File(...), background: bool = False):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if background:
        # Fire-and-forget is handled by caller polling /documents; here run inline
        # in a thread to keep code simple (FastAPI will still block briefly).
        pass
    try:
        result = ingest_single_file(dest, client=state["client"], embedder=state["embedder"])
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")
    return result


@app.delete("/documents/{source}")
def remove_document(source: str):
    try:
        delete_document(state["client"], source)
        return {"source": source, "status": "deleted"}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    from langchain_agent.rag import rag

    try:
        answer, context, retrieved = rag(
            req.query, top_k=req.top_k or settings.default_top_k,
            client=state["client"], embedder=state["embedder"],
            style=req.style,
        )
    except Exception as e:
        raise HTTPException(500, f"RAG failed: {e}")
    sources = [
        ChatSource(
            source=c.get("source"),
            headings=c.get("headings") or [],
            page_no=c.get("page_no"),
            score=c.get("score"),
            start=(c.get("chunk_text") or "")[:200],
        )
        for c in (retrieved or [])
    ]
    return ChatResponse(answer=str(answer), context=context or "", sources=sources)


@app.post("/voice")
async def voice(file: UploadFile = File(...), top_k: int = 5):
    """Audio upload -> transcript -> chat answer.

    Requires optional deps (faster-whisper + edge-tts). If missing, returns
    501 with install hint — text chat still works.
    """
    try:
        import edge_tts  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        raise HTTPException(
            501,
            "Voice needs `uv add faster-whisper edge-tts`. "
            "Meanwhile use /chat (text) or the LiveKit web call.",
        )
    from faster_whisper import WhisperModel

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path)
        transcript = " ".join(s.text for s in segments).strip()
        if not transcript:
            raise HTTPException(400, "Could not transcribe audio")
        from langchain_agent.rag import rag

        answer, context, retrieved = rag(
            transcript, top_k=top_k, client=state["client"],
            embedder=state["embedder"], style="voice",
        )
        # TTS to mp3 bytes
        import edge_tts

        tts_path = tmp_path + ".mp3"
        tts = edge_tts.Communicate(str(answer), voice="en-IN-NeerjaNeural")
        await tts.save(tts_path)
        from fastapi.responses import FileResponse

        return FileResponse(
            tts_path, media_type="audio/mpeg", filename="reply.mp3",
            headers={"X-Transcript": transcript[:500], "X-Answer": str(answer)[:1000]},
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/livekit/token")
def livekit_token(identity: str = "web-user", room: str = "nano-support"):
    """Mint a LiveKit token so the website can join the same voice agent."""
    try:
        from livekit import api as lk_api
    except ImportError:
        raise HTTPException(501, "livekit-api not installed; LiveKit console still works")
    token = (
        lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(lk_api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"url": settings.livekit_url, "token": token, "room": room}
