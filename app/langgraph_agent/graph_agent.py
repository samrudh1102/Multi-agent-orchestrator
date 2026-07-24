"""LangGraph agent path: a hand-built StateGraph that does RAG retrieval up
front, then lets the LLM decide (per turn) whether to call Jira/TestRail tools
or answer directly from the retrieved context.

Graph shape:

    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

The `agent` node runs the tool-bound Gemini chat model; `tools` is LangGraph's
prebuilt ToolNode executing the FastMCP-backed StructuredTools. The conditional
edge (tools_condition) routes to `tools` whenever the model emitted tool calls
and to END otherwise — i.e. the LLM itself decides whether tools are needed.

Only used when settings.enable_tools is true; selected in app/main.py."""
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.langgraph_agent.fastmcp_client import fastmcp_provider
from app.llm import _extract_text, get_tool_chat_model
from app.rag_pipeline import SYSTEM_PROMPT, build_user_prompt, retrieve_sources
from app.schemas import SourceChunk, ToolCallInfo

_graph = None  # compiled lazily on first use


def _build_graph():
    tools = fastmcp_provider.tools
    model = get_tool_chat_model()
    model_with_tools = model.bind_tools(tools) if tools else model

    def agent_node(state: MessagesState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")

    if tools:
        builder.add_node("tools", ToolNode(tools))
        # tools_condition -> "tools" when the last AIMessage has tool calls,
        # otherwise -> END.
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", END)

    return builder.compile()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def answer_question_with_graph(
    question: str, history: list[dict[str, str]] | None = None, top_k: int | None = None
) -> tuple[str, list[SourceChunk], list[ToolCallInfo]]:
    sources = retrieve_sources(question, top_k)
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
    # recursion_limit bounds the agent<->tools loop; each tool round trip is two
    # steps, so mirror the manual path's max_tool_iterations budget.
    result = await graph.ainvoke(
        {"messages": messages},
        config={"recursion_limit": settings.max_tool_iterations * 2 + 1},
    )

    out_messages = result["messages"]
    tool_call_records = _collect_tool_calls(out_messages)

    answer = ""
    for m in reversed(out_messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            answer = _extract_text(m.content)
            break

    return answer, sources, tool_call_records


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
