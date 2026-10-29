"""LangGraph agent path: a hand-built StateGraph that does RAG retrieval up
front, then lets the LLM decide (per turn) whether to call MCP tools or
answer directly from the retrieved context.

Graph shape (plain ReAct loop):

  START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

The `agent` node runs the tool-bound Gemini chat model; `tools` runs the
FastMCP-backed StructuredTools for whatever tool_calls the last AIMessage
carries, concurrently (see tools_node). The conditional edge (tools_condition)
routes to `tools` whenever the model emitted tool calls and to END otherwise.

Whether multiple tool_calls end up in one AIMessage (and therefore run
concurrently in `tools_node`) is up to Gemini deciding, turn by turn, that a
batch of calls is independent — SYSTEM_PROMPT (app/rag_pipeline.py) instructs
it to batch independent tool calls into the same turn rather than spreading
them one-per-turn.

Only used when settings.enable_tools is true; selected in app/main.py."""
import asyncio
import logging
import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import tools_condition

from app.config import settings
from app.langgraph_agent.fastmcp_client import fastmcp_provider
from app.llm import _extract_text, get_tool_chat_model
from app.rag_pipeline import SYSTEM_PROMPT, build_user_prompt, retrieve_sources
from app.schemas import SourceChunk, ToolCallInfo

logger = logging.getLogger(__name__)

_graph = None  # compiled lazily on first use

# Tool responses (ClickHouse log dumps, Postgres rows, S3 manifests, ...) can
# be huge; truncating keeps them from bloating the conversation history that
# gets re-sent to Gemini on every subsequent turn.
_MAX_TOOL_RESPONSE_CHARS = 4000
_TRUNCATION_NOTICE = "... [Response truncated to {} characters to preserve context limit]"

_TOOL_STATUS_LABELS = {
    "clickhouse_run_select_query": "📊 Querying ClickHouse logs & metrics...",
    "jira_searchJiraIssuesUsingJql": "🎫 Searching Jira defect tickets...",
    "databricks_execute_sql": "🔮 Querying Databricks ML predictions & health scores...",
    "airflow_list_dag_runs": "⚡ Checking Airflow pipeline execution status...",
    "postgres_query": "🐘 Querying PostgreSQL relational database...",
}
_TOOL_STATUS_PREFIXES = (
    ("testrail_", "📋 Fetching TestRail test run results..."),
    ("s3_", "📦 Inspecting S3 artifacts & video bundles..."),
)


def _tool_status_label(name: str) -> str:
    if name in _TOOL_STATUS_LABELS:
        return _TOOL_STATUS_LABELS[name]
    for prefix, label in _TOOL_STATUS_PREFIXES:
        if name.startswith(prefix):
            return label
    return f"Running {name}..."


def _debug(msg: str) -> None:
    """Verbose per-node/per-tool-call tracing, gated behind settings.debug_agent_logging
    since it fires on every node/tool call — noisy for normal runs but useful when
    diagnosing a specific graph run."""
    if settings.debug_agent_logging:
        print(msg)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _truncate(content: str) -> str:
    if len(content) <= _MAX_TOOL_RESPONSE_CHARS:
        return content
    notice = _TRUNCATION_NOTICE.format(_MAX_TOOL_RESPONSE_CHARS)
    return content[:_MAX_TOOL_RESPONSE_CHARS] + notice


def _make_agent_and_tools_nodes(tools_by_name: dict, bound_model):
    # The tool catalog is now a small, curated read-only allowlist (see
    # STRICT_TOOL_ALLOWLIST in fastmcp_client.py) rather than the full
    # ~170-tool catalog, so bind_tools() is cheap enough to call once — no
    # per-question server-subset selection or bound-model cache needed.
    async def agent_node(state: AgentState, config) -> dict:
        _debug(f"\n[AGENT] entering with {len(state['messages'])} message(s) in context")

        # on_delta (set via config["configurable"] by the /chat/stream path) lets
        # us push live text tokens to the SSE client. A turn that ends up calling
        # tools rarely carries visible text alongside the tool_call_chunks, but
        # to be safe we only start forwarding once the first chunk tells us which
        # kind of turn this is, and stick with that choice for the whole turn.
        on_delta = (config.get("configurable") or {}).get("on_delta") if config else None
        on_status = (config.get("configurable") or {}).get("on_status") if config else None

        t0 = time.monotonic()
        if on_delta is None:
            response = await bound_model.ainvoke(state["messages"])
        else:
            response = None
            forwarding = None
            async for chunk in bound_model.astream(state["messages"]):
                response = chunk if response is None else response + chunk
                if forwarding is None:
                    if chunk.tool_call_chunks:
                        forwarding = False
                    elif _extract_text(chunk.content):
                        forwarding = True
                if forwarding:
                    text = _extract_text(chunk.content)
                    if text:
                        await on_delta(text)
        logger.info("agent_node: LLM call took %.2fs", time.monotonic() - t0)
        if response.tool_calls:
            _debug(f"[AGENT] Gemini wants {len(response.tool_calls)} tool call(s): "
                  f"{[(c['name'], c['args']) for c in response.tool_calls]}")
            if on_status is not None:
                for call in response.tool_calls:
                    await on_status({
                        "stage": "tool_calling",
                        "name": call["name"],
                        "display": _tool_status_label(call["name"]),
                    })
        else:
            _debug(f"[AGENT] Gemini answered directly (no tool calls): {response.content!r:.200}")
        return {"messages": [response]}

    async def run_tool_call(call: dict) -> ToolMessage:
        tool = tools_by_name.get(call["name"])
        _debug(f"[TOOLS]   -> calling {call['name']}({call['args']})")
        t0 = time.monotonic()
        if tool is None:
            content = f"Unknown tool: {call['name']}"
            msg = ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])
        else:
            result = await tool.ainvoke({**call, "type": "tool_call"})
            msg = result if isinstance(result, ToolMessage) else ToolMessage(
                content=str(result), name=call["name"], tool_call_id=call["id"]
            )
            if isinstance(msg.content, str):
                msg.content = _truncate(msg.content)
        logger.info("tool_call: %s took %.2fs", call["name"], time.monotonic() - t0)
        _debug(f"[TOOLS]   <- {call['name']} done in {time.monotonic() - t0:.2f}s -> {str(msg.content)!r:.200}")
        return msg

    async def tools_node(state: AgentState, config) -> dict:
        """Same job as langgraph.prebuilt.ToolNode, but times each individual
        tool call and truncates oversized responses (see _truncate).

        When Gemini emits several tool_calls in one AIMessage it has already
        judged them independent (that's what "parallel function calling"
        means on its end) — nothing here re-checks for data dependencies
        between them, so run them concurrently rather than one Gemini round
        trip's worth of calls sequentially."""
        on_status = (config.get("configurable") or {}).get("on_status") if config else None
        calls = state["messages"][-1].tool_calls
        mode = "PARALLEL" if len(calls) > 1 else "SERIAL"
        _debug(f"[TOOLS] {mode} wave, running {len(calls)} call(s): {[c['name'] for c in calls]}")
        t0 = time.monotonic()
        outputs = await asyncio.gather(*(run_tool_call(call) for call in calls))
        logger.info(
            "tools_node: %s wave, %d call(s) [%s], wave took %.2fs",
            mode, len(calls), ", ".join(c["name"] for c in calls), time.monotonic() - t0,
        )
        _debug(f"[TOOLS] wave done in {time.monotonic() - t0:.2f}s")
        if on_status is not None:
            for call in calls:
                await on_status({
                    "stage": "tool_complete",
                    "name": call["name"],
                    "display": f"Finished {call['name']}",
                })
        return {"messages": list(outputs)}

    return agent_node, tools_node


def _build_graph():
    print("[FLOW] _build_graph: building graph (this only happens once, lazily)")
    # fastmcp_provider.tools is already filtered to the curated read-only
    # allowlist (see STRICT_TOOL_ALLOWLIST in fastmcp_client.py), so no
    # per-question tool selection or bound-model caching is needed here.
    all_tools = fastmcp_provider.tools
    print(f"[FLOW] _build_graph: {len(all_tools)} tool(s) available from fastmcp_provider")
    tools_by_name = {t.name: t for t in all_tools}
    model = get_tool_chat_model()

    bound_model = model
    if all_tools:
        # bind_tools() converts every tool's schema into Gemini's function-
        # declaration format. Doing that conversion once here, at startup,
        # instead of lazily on the first request, keeps that cost off the
        # first user-facing response — cheap now that the catalog is a
        # curated ~15 tools rather than the full ~170.
        t0 = time.monotonic()
        bound_model = model.bind_tools(all_tools)
        logger.info("_build_graph: bind_tools took %.2fs for %d tool(s)", time.monotonic() - t0, len(all_tools))

    agent_node, tools_node = _make_agent_and_tools_nodes(tools_by_name, bound_model)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)

    if not all_tools:
        print("[FLOW] _build_graph: no tools -> trivial graph START -> agent -> END")
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        return builder.compile()

    print("[FLOW] _build_graph: ReAct graph -> START -> agent <-> tools -> END")
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    # tools_condition -> "tools" whenever the last AIMessage has tool calls,
    # otherwise -> END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


def _get_graph():
    global _graph
    if _graph is None:
        print("[FLOW] _get_graph: no cached graph, building now")
        _graph = _build_graph()
    else:
        print("[FLOW] _get_graph: reusing cached graph")
    return _graph


def warm_up() -> None:
    """Force the graph to build (and its full-set bind_tools to run) now,
    instead of lazily on the first request. Call once at app startup, after
    fastmcp_provider.connect() has populated the tool list."""
    print("[FLOW] warm_up: forcing graph build at startup")
    _get_graph()



# async def answer_question_with_graph_stream(
#     question: str, history: list[dict[str, str]] | None = None, top_k: int | None = None
# ):
#     # Only retrieve from FAISS if explicitly needed; avoid 2-second embedding penalty on live SQL queries
#     sources = []
#     # If question mentions policy/handbook/docs, you can optionally run retrieve_sources
#     yield ("sources", sources)

#     messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
#     for turn in history or []:
#         role = turn.get("role")
#         content = turn.get("content", "")
#         if role == "user":
#             messages.append(HumanMessage(content=content))
#         elif role == "assistant":
#             messages.append(AIMessage(content=content))
#     messages.append(HumanMessage(content=question))

#     graph = _get_graph()
#     queue: asyncio.Queue = asyncio.Queue()

#     async def on_delta(text: str) -> None:
#         await queue.put(("delta", text))

#     async def on_status(status: dict) -> None:
#         await queue.put(("status", status))

#     async def run_graph():
#         try:
#             return await graph.ainvoke(
#                 {"messages": messages},
#                 config={
#                     "configurable": {"on_delta": on_delta, "on_status": on_status},
#                     "recursion_limit": settings.max_tool_iterations * 2 + 1,
#                 },
#             )
#         finally:
#             await queue.put(None)

#     task = asyncio.create_task(run_graph())

#     while True:
#         item = await queue.get()
#         if item is None:
#             break
#         kind, payload = item
#         yield (kind, payload)

#     result = await task
#     tool_call_records = _collect_tool_calls(result["messages"])
#     yield ("tool_calls", tool_call_records)
async def answer_question_with_graph_stream(
    question: str, history: list[dict[str, str]] | None = None, top_k: int | None = None
):
    """Streaming variant of answer_question_with_graph.

    Yields ("sources", list[SourceChunk]) immediately, then ("status", dict)
    each time a tool call starts/completes, interleaved with ("delta", str)
    for each text token as the agent's final (non-tool-calling) turn streams
    from Gemini, then ("tool_calls", list[ToolCallInfo]) once the graph
    finishes — tool_calls aren't known until the graph completes, since
    they're the result of tool execution during the run, so they necessarily
    come last.
    """
    print(f"[FLOW] answer_question_with_graph_stream: called with question={question!r}, "
          f"history has {len(history or [])} turn(s)")
    t0 = time.monotonic()
    # retrieve_sources is a blocking call (embedding + FAISS); offload it so
    # it doesn't stall the event loop, mirroring the non-tools path in main.py.
    sources = await asyncio.to_thread(retrieve_sources, question, top_k)
    retrieve_elapsed = time.monotonic() - t0
    logger.info("answer_question_with_graph_stream: retrieve_sources took %.2fs", retrieve_elapsed)
    print(f"[FLOW] answer_question_with_graph_stream: retrieved {len(sources)} source(s) in "
          f"{retrieve_elapsed:.2f}s, yielding ('sources', ...)")
    yield ("sources", sources)

    user_prompt = build_user_prompt(question, sources, allow_tools=True)

    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_prompt))

    graph = _get_graph()
    queue: asyncio.Queue = asyncio.Queue()

    async def on_delta(text: str) -> None:
        await queue.put(("delta", text))

    async def on_status(status: dict) -> None:
        await queue.put(("status", status))

    async def run_graph():
        try:
            return await graph.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {"on_delta": on_delta, "on_status": on_status},
                    "recursion_limit": settings.max_tool_iterations * 2 + 1,
                },
            )
        finally:
            await queue.put(None)  # sentinel: no more events coming

    task = asyncio.create_task(run_graph())
    print(f"[FLOW] answer_question_with_graph_stream: graph.ainvoke started as background task, "
          f"streaming status/deltas as they arrive")
    t1 = time.monotonic()
    delta_count = 0
    while True:
        item = await queue.get()
        if item is None:
            break
        kind, payload = item
        if kind == "delta":
            delta_count += 1
        yield (kind, payload)
    graph_elapsed = time.monotonic() - t1
    print(f"[FLOW] answer_question_with_graph_stream: delta stream done, {delta_count} delta(s) "
          f"yielded in {graph_elapsed:.2f}s")

    result = await task
    tool_call_records = _collect_tool_calls(result["messages"])
    print(f"[FLOW] answer_question_with_graph_stream: graph task finished, "
          f"{len(tool_call_records)} tool call record(s), yielding ('tool_calls', ...)")
    yield ("tool_calls", tool_call_records)

    total_elapsed = time.monotonic() - t0
    logger.info(
        "answer_question_with_graph_stream: TOTAL %.2fs (retrieve_sources %.2fs + graph %.2fs)",
        total_elapsed, retrieve_elapsed, graph_elapsed,
    )
    print(f"[FLOW] answer_question_with_graph_stream: TOTAL {total_elapsed:.2f}s "
          f"(retrieve {retrieve_elapsed:.2f}s + graph {graph_elapsed:.2f}s)")


def _collect_tool_calls(messages: list) -> list[ToolCallInfo]:
    """Pair each AIMessage tool call with its ToolMessage result for the API
    response, so the UI can show what the agent actually invoked."""
    args_by_id: dict[str, tuple[str, dict]] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                args_by_id[tc["id"]] = (tc["name"], tc["args"])

    records: list[ToolCallInfo] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            name, args = args_by_id.get(m.tool_call_id, (m.name or "unknown", {}))
            content = m.content if isinstance(m.content, str) else str(m.content)
            records.append(ToolCallInfo(name=name, arguments=args, result=content))
    return records
