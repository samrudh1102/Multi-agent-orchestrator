"""Streamlit frontend for the RAG Chatbot API.

Talks to the FastAPI backend (see app/main.py) over HTTP:
  POST   /chat                     -> answer + sources (+ tool_calls when ENABLE_TOOLS=true)
  POST   /documents/upload         -> ingest a .pdf/.docx/.txt/.md file
  GET    /documents                -> list ingested documents
  DELETE /documents/{doc_id}       -> remove a document from the index
  GET    /health                   -> liveness check

Run the backend first (uvicorn app.main:app --port 8000), then:
    streamlit run app/streamlit_app.py
"""

import json
import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 1000  # generous: embedding + LLM (+ possible tool calls) can be slow

st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_url(path: str) -> str:
    base = st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")
    return f"{base}{path}"


def check_health() -> bool:
    try:
        r = requests.get(api_url("/health"), timeout=5)
        return r.ok and r.json().get("status") == "ok"
    except requests.RequestException:
        return False


def list_documents() -> list[dict]:
    r = requests.get(api_url("/documents"), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def upload_document(file) -> dict:
    files = {"file": (file.name, file.getvalue())}
    r = requests.post(api_url("/documents/upload"), files=files, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def delete_document(doc_id: str) -> None:
    r = requests.delete(api_url(f"/documents/{doc_id}"), timeout=REQUEST_TIMEOUT)
    r.raise_for_status()


def send_chat(question: str, history: list[dict]) -> dict:
    payload = {"question": question, "history": history}
    top_k = st.session_state.get("top_k")
    if top_k:
        payload["top_k"] = top_k
    r = requests.post(api_url("/chat"), json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def stream_chat(question: str, history: list[dict], captured: dict):
    """Open the /chat/stream SSE endpoint and yield answer text deltas.

    Sources and tool_calls arrive as their own SSE events; we stash them into
    `captured` (mutated in place) so the caller can render them once the text
    stream — driven by st.write_stream over this generator — is exhausted.
    """
    payload = {"question": question, "history": history}
    top_k = st.session_state.get("top_k")
    if top_k:
        payload["top_k"] = top_k

    with requests.post(
        api_url("/chat/stream"), json=payload, stream=True, timeout=REQUEST_TIMEOUT
    ) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:  # blank line terminates an SSE frame
                event = None
                continue
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                if event == "delta":
                    yield data.get("text", "")
                elif event == "sources":
                    captured["sources"] = data
                elif event == "tool_calls":
                    captured["tool_calls"] = data
                elif event == "error":
                    raise RuntimeError(data.get("detail", "stream error"))


def error_detail(exc: requests.RequestException) -> str:
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return resp.json().get("detail", resp.text)
        except ValueError:
            return resp.text or str(exc)
    return str(exc)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**{i}. {s.get('filename', '?')}** "
                f"· chunk {s.get('chunk_index', '?')} "
                f"· score {s.get('score', 0):.3f}"
            )
            st.caption(s.get("text", ""))
            if i < len(sources):
                st.divider()


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🛠️ Tool calls ({len(tool_calls)})"):
        for i, t in enumerate(tool_calls, 1):
            st.markdown(f"**{i}. `{t.get('name', '?')}`**")
            st.caption("Arguments")
            st.json(t.get("arguments", {}))
            st.caption("Result")
            st.code(str(t.get("result", "")))
            if i < len(tool_calls):
                st.divider()


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    # Each message: {"role", "content", "sources"?, "tool_calls"?}
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# --------------------------------------------------------------------------- #
# Sidebar: connection, documents, settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.top_k = st.number_input(
        "top_k (0 = backend default)", min_value=0, max_value=50, value=0, step=1
    ) or None
    st.session_state.stream = st.toggle(
        "Stream responses", value=st.session_state.get("stream", True)
    )

    healthy = check_health()
    st.markdown(
        f"**Backend:** {'🟢 connected' if healthy else '🔴 unreachable'}"
    )

    st.divider()
    st.header("📄 Documents")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=False,
    )
    if uploaded is not None and st.button("Ingest", use_container_width=True):
        with st.spinner(f"Ingesting {uploaded.name}…"):
            try:
                res = upload_document(uploaded)
                st.success(f"Added {res['filename']} ({res['num_chunks']} chunks)")
            except requests.RequestException as e:
                st.error(f"Upload failed: {error_detail(e)}")

    if healthy:
        try:
            docs = list_documents()
        except requests.RequestException as e:
            docs = []
            st.error(f"Couldn't list documents: {error_detail(e)}")

        if not docs:
            st.caption("No documents ingested yet.")
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{d['filename']}**  \n`{d['num_chunks']} chunks`")
            if col2.button("🗑️", key=f"del_{d['doc_id']}", help="Delete"):
                try:
                    delete_document(d["doc_id"])
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Delete failed: {error_detail(e)}")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main: chat
# --------------------------------------------------------------------------- #
st.title("💬 AI Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_tool_calls(msg.get("tool_calls", []))
        render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # History = prior turns only (role/content), excluding the message we just added.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        try:
            if st.session_state.get("stream", True):
                captured = {"sources": [], "tool_calls": []}
                answer = st.write_stream(stream_chat(prompt, history, captured))
                sources = captured["sources"]
                tool_calls = captured["tool_calls"]
            else:
                with st.spinner("Thinking…"):
                    data = send_chat(prompt, history)
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                tool_calls = data.get("tool_calls", [])
                st.markdown(answer)
        except (requests.RequestException, RuntimeError) as e:
            detail = error_detail(e) if isinstance(e, requests.RequestException) else str(e)
            err = f"⚠️ Request failed: {detail}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            render_tool_calls(tool_calls)
            render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tool_calls": tool_calls,
                }
            )
