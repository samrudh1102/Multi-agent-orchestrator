import truststore
truststore.inject_into_ssl()

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import traceback
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import rag_pipeline
from app.config import settings
from app.document_loader import UnsupportedFileTypeError
from app.langgraph_agent import graph_agent
from app.langgraph_agent.fastmcp_client import fastmcp_provider
from app.schemas import ChatRequest, ChatResponse, DocumentInfo, UploadResponse

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_tools:
        await fastmcp_provider.connect()
        # Pay the multi-second bind_tools() cost (see graph_agent.warm_up)
        # now, at startup, instead of on whichever request happens to hit it
        # first.
        await asyncio.to_thread(graph_agent.warm_up)
    yield
    if settings.enable_tools:
        await fastmcp_provider.disconnect()


app = FastAPI(
    title="RAG Chatbot API",
    description="A retrieval-augmented chatbot backend using Google Gemini and FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    temp_path = settings.upload_path / f"{uuid.uuid4()}{suffix}"
    try:
        with temp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        doc_id, num_chunks = rag_pipeline.ingest_document(temp_path, file.filename)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        temp_path.unlink(missing_ok=True)

    return UploadResponse(doc_id=doc_id, filename=file.filename, num_chunks=num_chunks)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in rag_pipeline.list_documents()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    deleted = rag_pipeline.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "doc_id": doc_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if settings.enable_tools:
            answer, sources, tool_calls = await graph_agent.answer_question_with_graph(
                request.question, request.history, request.top_k
            )
        else:
            # rag_pipeline.answer_question is a blocking call (embedding +
            # FAISS + HTTP); offload it so it doesn't stall the event loop.
            answer, sources = await asyncio.to_thread(
                rag_pipeline.answer_question, request.question, request.history, request.top_k
            )
            tool_calls = []
    except Exception as e:
        print("=" * 80)
        print(type(e))
        print(repr(e))
        traceback.print_exc()
        print("=" * 80)
        raise
    return ChatResponse(answer=answer, sources=sources, tool_calls=tool_calls)


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _dump(obj):
    """Serialize a pydantic model (SourceChunk/ToolCallInfo) or pass through dicts."""
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _rag_event_stream(request: ChatRequest):
    """Sync generator (RAG path): emit sources, then answer deltas, then done.

    Starlette runs sync generators in a threadpool, so the blocking embedding +
    Gemini streaming calls here don't stall the event loop.
    """
    try:
        sources, deltas = rag_pipeline.answer_question_stream(
            request.question, request.history, request.top_k
        )
        yield _sse("sources", [_dump(s) for s in sources])
        for text in deltas:
            yield _sse("delta", {"text": text})
        yield _sse("done", {})
    except Exception as e:  # e.g. missing/invalid GOOGLE_API_KEY, Gemini quota
        logger.exception("rag chat stream failed")
        yield _sse("error", {"detail": str(e)})


async def _tools_event_stream(request: ChatRequest):
    """Async generator (ENABLE_TOOLS path): streams tool-execution `status`
    events as the agent invokes tools, then the agent's final answer turn
    token-by-token as Gemini generates it. tool_calls aren't known until the
    graph finishes (they're the result of tool execution during the run), so
    unlike the RAG path, that event necessarily arrives after the deltas."""
    try:
        async for event, payload in graph_agent.answer_question_with_graph_stream(
            request.question, request.history, request.top_k
        ):
            if event == "sources":
                yield _sse("sources", [_dump(s) for s in payload])
            elif event == "status":
                yield _sse("status", payload)
            elif event == "delta":
                yield _sse("delta", {"text": payload})
            elif event == "tool_calls":
                yield _sse("tool_calls", [_dump(t) for t in payload])
        yield _sse("done", {})
    except Exception as e:
        logger.exception("tools chat stream failed")
        yield _sse("error", {"detail": str(e)})


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming variant of /chat over Server-Sent Events.

    Events: `sources` (list), then `status` (dict, tools mode only, repeated
    as tools run), `delta` ({"text": ...}) repeated, `tool_calls` (list,
    tools mode only), then `done` — or a single `error` ({"detail": ...}).
    """
    generator = (
        _tools_event_stream(request)
        if settings.enable_tools
        else _rag_event_stream(request)
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/tools")
async def list_tools() -> list[dict]:
    """Debug endpoint: shows which MCP tools are currently connected/available."""
    if not settings.enable_tools:
        raise HTTPException(status_code=400, detail="ENABLE_TOOLS is not set to true")
    return [
        {"name": t.name, "description": t.description, "parameters": t.args_schema}
        for t in fastmcp_provider.tools
    ]
