# RAG Chatbot API (FastAPI + Google Gemini + FAISS)

A retrieval-augmented generation (RAG) chatbot backend:

- **FastAPI** for the HTTP API
- **Google Gemini** for both embeddings (`text-embedding-004`) and text generation
- **FAISS** (in-memory, file-persisted) for vector search
- Ingests **PDF, DOCX, TXT, and Markdown** files

## Project layout

```
rag-chatbot/
├── app/
│   ├── main.py            # FastAPI app + routes
│   ├── config.py          # Settings (env vars)
│   ├── schemas.py          # Pydantic request/response models
│   ├── document_loader.py # PDF/DOCX/TXT/MD -> plain text
│   ├── text_splitter.py    # Chunking with overlap
│   ├── embeddings.py       # Google Gemini embedding wrapper
│   ├── vector_store.py     # FAISS index + metadata, with persistence
│   ├── llm.py              # Google Gemini chat model (plain + tool-calling)
│   ├── rag_pipeline.py     # Ties ingestion + retrieval + generation together
│   └── langgraph_agent/    # tool-calling path (ENABLE_TOOLS=true): FastMCP + LangGraph
│       ├── fastmcp_client.py  # FastMCP client -> MCP tools as LangChain tools
│       └── graph_agent.py     # LangGraph StateGraph agent (RAG + MCP tools)
├── scripts/
│   └── setup_jira_auth.py # One-time interactive OAuth login for Jira's remote MCP server
├── data/
│   ├── uploads/            # temp storage during ingestion
│   ├── index/              # persisted FAISS index + metadata.json
│   └── mcp_tokens/         # persisted Jira OAuth tokens (created by setup_jira_auth.py)
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then create a `.env` file in the project root (it's git-ignored — never commit real keys) with the variables below.

### Environment variables

**Gemini (required)** — set `GOOGLE_API_KEY` to a key from [Google AI Studio](https://aistudio.google.com/apikey). Both embeddings and generation go through the Gemini API — no local GPU, model downloads, or separate local server needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | _(required)_ | Gemini API key from Google AI Studio |
| `EMBEDDING_MODEL_NAME` | `models/text-embedding-004` | Gemini embedding model |
| `LLM_MODEL_NAME` | `gemini-2.5-flash` | Model used to generate RAG answers |
| `TOOL_MODEL_NAME` | `gemini-2.5-flash` | Model used for tool-calling (see below) |

**Chunking / retrieval / generation**

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between adjacent chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `MAX_NEW_TOKENS` | `1545` | Max tokens in a generated answer |
| `TEMPERATURE` | `0.3` | Generation temperature |
| `UPLOAD_DIR` | `data/uploads` | Temp storage during ingestion |
| `INDEX_DIR` | `data/index` | Persisted FAISS index + metadata |

**MCP tools** (all optional; only active when `ENABLE_TOOLS=true`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TOOLS` | `false` | Master switch for Jira/TestRail/etc. tool calling |
| `MAX_TOOL_ITERATIONS` | `5` | Max tool-call rounds per chat turn |
| `JIRA_MCP_URL` | Atlassian hosted URL | Remote Jira MCP endpoint (see [Jira setup](#jira-setup)) |
| `JIRA_TOKEN_PATH` | `data/mcp_tokens/jira.json` | Where OAuth tokens are stored |
| `TESTRAIL_URL` / `TESTRAIL_USERNAME` / `TESTRAIL_API_KEY` | — | TestRail credentials (see [TestRail setup](#testrail-setup)) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | — | AWS creds for S3 / Redshift tools |
| `REDSHIFT_DATABASE` / `REDSHIFT_WORKGROUP_NAME` | — | Redshift Serverless target |
| `CLICKHOUSE_URL` / `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` / `CLICKHOUSE_DATABASE` | — | ClickHouse REST connection |
| `AIRFLOW_URL` / `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | — | Airflow connection |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `DATABRICKS_WAREHOUSE_ID` | — | Databricks SQL warehouse |

## Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## API

### `POST /documents/upload`
Upload a `.pdf`, `.docx`, `.txt`, or `.md` file (multipart form, field name `file`). Extracts text, chunks it, embeds it, and adds it to the FAISS index.

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/documents/upload
```
```json
{ "doc_id": "…", "filename": "handbook.pdf", "num_chunks": 42 }
```

### `GET /documents`
List ingested documents and their chunk counts.

### `DELETE /documents/{doc_id}`
Removes a document's chunks from the index (rebuilds the FAISS index without them).

### `POST /chat`
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?"}'
```
```json
{
  "answer": "...",
  "sources": [
    { "doc_id": "…", "filename": "handbook.pdf", "chunk_index": 3, "text": "...", "score": 0.82 }
  ]
}
```

Optional fields on the request:
- `top_k`: override how many chunks to retrieve (default from `.env`)
- `history`: prior turns as `[{"role": "user"|"assistant", "content": "..."}]` for multi-turn context

When `ENABLE_TOOLS=true`, the response also includes a `tool_calls` list (name, arguments, result) for any Jira/TestRail MCP tools the model invoked while answering — see below.

### `GET /tools`
Debug endpoint (only enabled when `ENABLE_TOOLS=true`): lists the MCP tools currently available, namespaced by server (`jira_...`, `testrail_...`). Useful for confirming what's actually connected without going through `/chat`.

### `GET /health`
Basic liveness check.

## MCP tool integration (Jira + TestRail)

Beyond document RAG, the chatbot can call out to live Jira and TestRail data via [MCP](https://modelcontextprotocol.io) tools, gated behind `ENABLE_TOOLS` in `.env`. This is additive — document retrieval keeps working as before, and tool-calling is implemented against Gemini.

The implementation (`app/langgraph_agent/`) retrieves RAG context first, then lets the LLM decide per turn whether to call a Jira/TestRail tool or answer directly from context. A [FastMCP](https://gofastmcp.com) client (`fastmcp_client.py`) connects to both servers and exposes their tools as LangChain tools, driven by a LangGraph `StateGraph` (`graph_agent.py`): `START → agent → tools_condition → tools → agent → … → END`. The `agent` node runs `get_tool_chat_model()` (`app/llm.py`), a Gemini chat model bound to the MCP tools via `bind_tools()`. Jira auth reuses the token minted by `setup_jira_auth.py` as a bearer header.

### Why a separate tool-calling model

`TOOL_MODEL_NAME` in `.env` lets you point tool-calling at a different Gemini model (e.g. a larger/smaller one) independently of the RAG-answer model in `LLM_MODEL_NAME`.

### Jira setup

Jira is reached through [Atlassian's official hosted remote MCP server](https://www.atlassian.com/platform/remote-mcp-server) (OAuth 2.1, covers Jira/Confluence/Bitbucket). One-time setup:

```bash
python scripts/setup_jira_auth.py
```

This opens your browser for the Atlassian consent screen and saves the resulting tokens to `JIRA_TOKEN_PATH` (default `data/mcp_tokens/jira.json`). The running app only loads and refreshes these tokens — it never does the interactive login itself, so re-run this script if the refresh token is ever revoked.

### TestRail setup

TestRail has no official MCP server, so this repo ships its **own** one built with [FastMCP](https://gofastmcp.com): `app/testrail/mcp_server.py`. It runs as a local stdio subprocess and talks to TestRail's REST API through `app/testrail/` (`client.py` → `service.py`), exposing 24 tools — browse projects/suites/sections/cases/runs/tests/results and create cases/runs/results/milestones/sections, plus reference lookups (`get_statuses`, `get_priorities`, `get_case_types`).

By default the app launches it with the current Python interpreter (see `app/config.py`), so there's nothing extra to install or start — just provide credentials in `.env`:

```
TESTRAIL_URL=https://yourcompany.testrail.io
TESTRAIL_USERNAME=you@yourcompany.com
TESTRAIL_API_KEY=your-api-key
```

`TESTRAIL_API_KEY` is an API key from **My Settings → API Keys** in TestRail (or your account password if API keys are disabled). TestRail tools activate only when all three values are set and `ENABLE_TOOLS=true`. To launch the server differently (e.g. a wrapper script), override `TESTRAIL_MCP_COMMAND` / `TESTRAIL_MCP_ARGS`; leave them unset to use the default.

Smoke-test the server on its own:

```bash
TESTRAIL_URL=... TESTRAIL_USERNAME=... TESTRAIL_API_KEY=... \
    python app/testrail/mcp_server.py
```

### Known limitation: `--reload`

`uvicorn --reload` re-execs the app process on file changes. With `ENABLE_TOOLS=true`, that means a fresh TestRail subprocess and a fresh Jira session get spun up on every reload, leaking the old ones. Run without `--reload` once tools are enabled:

```bash
uvicorn app.main:app --port 8000
```

## How it works

1. **Ingestion**: file → extracted text (`document_loader.py`) → chunked with overlap (`text_splitter.py`) → embedded (`embeddings.py`) → stored in FAISS + a parallel JSON metadata store (`vector_store.py`), persisted to `data/index/`.
2. **Query**: question embedded → FAISS cosine-similarity search (`IndexFlatIP` on normalized vectors) → top-k chunks assembled into a context block → sent to the LLM with a system prompt instructing it to answer only from context (`rag_pipeline.py`) → response + cited source chunks returned.

## Notes & things you may want to change

- **Persistence**: the FAISS index is a single flat index rebuilt on delete — fine for thousands of chunks, but for very large corpora you'll want an approximate index (e.g. `IndexIVFFlat`) and a real database instead of the JSON metadata file.
- **Concurrency**: a simple `threading.Lock` guards the in-process index. For multiple worker processes, you'd want a shared vector DB (e.g. Qdrant, Weaviate, pgvector) instead of a local FAISS file.
- **Auth**: there's no authentication on the endpoints — add an API key or OAuth dependency before exposing this publicly.
- **Model choice**: swap `LLM_MODEL_NAME` / `EMBEDDING_MODEL_NAME` freely for any Gemini generation/embedding model.
