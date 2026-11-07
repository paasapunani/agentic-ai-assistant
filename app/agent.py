"""LangGraph agent: planner -> agent <-> tools loop, with per-thread memory."""
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import settings
from .tools import TOOLS


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation state
    plan: str                                # latest task plan


def get_llm():
    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.anthropic_model, temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=settings.openai_model, temperature=0)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
    )


TOOL_DOCS = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS)

PLANNER_PROMPT = f"""You are a planning module for an AI assistant.
Available tools:
{TOOL_DOCS}

Write a short numbered plan (max 5 steps) for answering the user's latest
message. If no tools are needed, reply with: 1. Answer directly."""

AGENT_PROMPT = """You are a helpful assistant that can call tools.
Follow the plan, call tools when needed (several if required), and then give a
clear final answer. Never invent tool results.

Current plan:
{plan}"""


def build_graph():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def planner(state: AgentState):
        msgs = [SystemMessage(content=PLANNER_PROMPT)] + state["messages"][-6:]
        plan = llm.invoke(msgs)
        return {"plan": _text(plan.content)}

    def agent(state: AgentState):
        system = SystemMessage(content=AGENT_PROMPT.format(plan=state.get("plan", "")))
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_agent():
    return build_graph()


def run_agent(message: str, thread_id: str = "default") -> dict:
    agent = get_agent()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config)

    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    steps = []
    for m in msgs[last_human + 1:]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                steps.append({"type": "tool_call", "name": call["name"], "args": call["args"]})
        elif isinstance(m, ToolMessage):
            steps.append({"type": "tool_result", "name": m.name, "content": _text(m.content)})

    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {
        "reply": _text(final.content) if final else "",
        "plan": result.get("plan", ""),
        "steps": steps,
    }
