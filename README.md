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
│   ├── setup_jira_auth.py       # One-time interactive OAuth login for Jira's remote MCP server
│   └── setup_clickhouse_auth.py # One-time interactive OAuth login for ClickHouse Cloud's remote MCP server
├── data/
│   ├── uploads/            # temp storage during ingestion
│   ├── index/              # persisted FAISS index + metadata.json
│   └── mcp_tokens/         # persisted Jira/ClickHouse OAuth tokens (created by the setup_*_auth.py scripts)
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
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | — | AWS creds for S3 (via AWS's remote MCP Server) and Redshift tools |
| `AWS_MCP_ENDPOINT` | `https://aws-mcp.us-east-1.api.aws/mcp` | AWS's official remote MCP Server endpoint (see [S3 setup](#s3-setup)) |
| `AWS_MCP_REGION` | `AWS_DEFAULT_REGION` | Region passed to `mcp-proxy-for-aws`; separate from `AWS_DEFAULT_REGION` in case your S3 buckets live in a different region than the MCP endpoint |
| `AWS_MCP_READ_ONLY` | `true` | Disable write-capable AWS tools in `mcp-proxy-for-aws` |
| `REDSHIFT_DATABASE` / `REDSHIFT_WORKGROUP_NAME` | — | Redshift Serverless target |
| `CLICKHOUSE_URL` / `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` / `CLICKHOUSE_DATABASE` | — | ClickHouse Cloud REST connection; no longer used for MCP, only by `scripts/seed_jira_mock.py` |
| `CLICKHOUSE_MCP_URL` | `https://mcp.clickhouse.cloud/mcp` | ClickHouse Cloud's official remote MCP Server endpoint (see [ClickHouse setup](#clickhouse-setup)) |
| `CLICKHOUSE_TOKEN_PATH` | `data/mcp_tokens/clickhouse.json` | Where ClickHouse Cloud OAuth tokens are stored |
| `AIRFLOW_URL` / `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | — | Airflow connection |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | — | Databricks workspace + PAT for Databricks' remote MCP Server (see [Databricks setup](#databricks-setup)) |

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

## MCP tool integration (Jira, TestRail, Databricks, S3, ClickHouse, ...)

Beyond document RAG, the chatbot can call out to live Jira, TestRail, Databricks, S3, ClickHouse and other data via [MCP](https://modelcontextprotocol.io) tools, gated behind `ENABLE_TOOLS` in `.env`. This is additive — document retrieval keeps working as before, and tool-calling is implemented against Gemini.

The implementation (`app/langgraph_agent/`) retrieves RAG context first, then lets the LLM decide per turn whether to call an MCP tool or answer directly from context. A [FastMCP](https://gofastmcp.com)/`langchain-mcp-adapters` client (`fastmcp_client.py`) connects to all configured servers and exposes their tools as LangChain tools, driven by a LangGraph `StateGraph` (`graph_agent.py`): `START → agent → tools_condition → tools → agent → … → END`. The `agent` node runs `get_tool_chat_model()` (`app/llm.py`), a Gemini chat model bound to the MCP tools via `bind_tools()`. Jira, ClickHouse and Databricks auth are sent as bearer headers/OAuth; S3 auth is handled by AWS's own signing proxy (see below).

**Databricks, S3 and ClickHouse go through their vendors' official remote MCP servers, not this repo's own local ones.** `app/databricks/mcp_server.py`, `app/s3/mcp_server.py` and `app/clickhouse/mcp_server.py` still exist in the repo (kept for reference/learning) but are no longer wired up in `app/config.py` / `fastmcp_client.py` — see [Databricks setup](#databricks-setup), [S3 setup](#s3-setup) and [ClickHouse setup](#clickhouse-setup) below for the replacements.

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

### Databricks setup

Databricks tools go through [Databricks' own managed remote MCP server](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp) for SQL Warehouses, at `https://<your-workspace-host>/api/2.0/mcp/sql`, reached directly over streamable HTTP — no local process is started. It exposes `execute_sql`/`poll_response` tools (async SQL execution against your default SQL Warehouse). Auth reuses the same personal access token as before, sent as `Authorization: Bearer <token>`:

```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-pat
```

Databricks tools activate when both are set and `ENABLE_TOOLS=true`. Note the previous local server (`app/databricks/mcp_server.py`, kept in-repo for reference) also exposed Unity Catalog browsing (`list_catalogs`/`list_schemas`/`list_tables`) and Jobs API tools (`list_jobs`/`trigger_job`) — Databricks' managed MCP endpoint doesn't have a remote equivalent for these, so only SQL querying is available now. `DATABRICKS_WAREHOUSE_ID` is no longer used by the client; the managed endpoint picks your SQL Warehouse server-side (configurable as a per-user default warehouse in the Databricks workspace UI).

### S3 setup

S3 tools go through [AWS's official remote MCP Server](https://aws.amazon.com/about-aws/whats-new/2026/05/aws-mcp-server/) (GA May 2026), a single managed endpoint covering all AWS APIs (not just S3) via one generic `call_aws` tool. Because that server authenticates with IAM SigV4 rather than the OAuth the MCP spec expects, AWS ships [`mcp-proxy-for-aws`](https://github.com/aws/mcp-proxy-for-aws), an official local bridge process that signs requests with your existing AWS credentials and forwards them to the remote server. It's launched automatically via `uvx` — this replaces `app/s3/mcp_server.py` (kept in-repo for reference), so no S3 logic runs locally anymore. Requires `uv`/`uvx` on PATH (`pip install uv` or see [astral.sh/uv](https://astral.sh/uv)).

Auth reuses the same AWS credentials as before:

```
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

S3 tools activate when the two AWS credential vars above are set and `ENABLE_TOOLS=true`. Optional overrides: `AWS_MCP_ENDPOINT` (defaults to `https://aws-mcp.us-east-1.api.aws/mcp`; AWS also runs one in `eu-central-1`), `AWS_MCP_REGION` (region passed to the proxy, defaults to `AWS_DEFAULT_REGION`), `AWS_MCP_READ_ONLY` (defaults to `true`, matching the read-only scope of the old `list_buckets`/`list_objects`/`get_object` tools). Note the tool surface changes from those S3-specific tool names to a single generic `call_aws` tool the model uses to invoke any AWS API action — not S3-only anymore, so IAM permissions should be scoped accordingly if you want to keep access limited to S3.

### ClickHouse setup

ClickHouse tools go through [ClickHouse Cloud's own remote MCP server](https://clickhouse.com/docs/cloud/features/ai-ml/remote-mcp) (currently in private preview) at a single global endpoint, reached directly over streamable HTTP — no local process is started. It must be enabled per-service first: in the ClickHouse Cloud console, open your service, click **Connect**, select **MCP**, and enable it. It exposes `run_select_query`, `list_databases` and `list_tables` (with column definitions) covering the same ground as the old local tools, plus extra service/backup/billing management tools; there's no exact `describe_table` equivalent, though `list_tables`' column definitions cover most of that need.

Unlike Databricks (PAT) or your existing `CLICKHOUSE_USER`/`CLICKHOUSE_PASSWORD`, ClickHouse Cloud's remote MCP server **only supports interactive OAuth 2.0** — there's no API key option. So, same as Jira, one-time setup:

```bash
python scripts/setup_clickhouse_auth.py
```

This opens your browser for the ClickHouse Cloud consent screen and saves the resulting tokens to `CLICKHOUSE_TOKEN_PATH` (default `data/mcp_tokens/clickhouse.json`). The running app only loads and refreshes these tokens — it never does the interactive login itself, so re-run this script if the refresh token is ever revoked. Access is scoped to whatever organizations/services your ClickHouse Cloud login can see, so `CLICKHOUSE_URL`/`CLICKHOUSE_USER`/`CLICKHOUSE_PASSWORD`/`CLICKHOUSE_DATABASE` are no longer used for MCP (they're still used directly by `scripts/seed_jira_mock.py`).

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
