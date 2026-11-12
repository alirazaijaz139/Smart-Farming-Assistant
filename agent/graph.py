import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.config import GROQ_API_KEY, GROQ_MODEL

from .tools import get_weather, recommend_crop, search_docs

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=GROQ_MODEL, api_key=GROQ_API_KEY,
            temperature=0, timeout=60, max_retries=3,
        )
    return _llm


class AgentState(TypedDict, total=False):
    question: str
    intent: str
    params: dict
    tool_result: dict
    answer: str


ROUTE_PROMPT = """You route questions for a farming assistant. Return ONLY a JSON object:
{"intent": "crop" | "weather" | "knowledge" | "chat", "params": {...}}
- crop: user wants a crop recommended. params: N, P, K, temperature, humidity, ph, rainfall
  (numbers; use null for any value the user did not give).
- weather: user asks about weather. params: {"city": "<city name>"}
- knowledge: farming how-to questions (sowing, irrigation, pests, fertilizer). params: {}
- chat: greetings or anything else. params: {}"""

ANSWER_PROMPT = """You are a helpful farming assistant. Use ONLY the tool result below.
If it has an error or missing values, tell the user exactly what is missing.
If it has document chunks, answer only from them and say you don't know if they don't cover the question.
Keep it short and practical.Never mention "the tool" or "tool result". If values are missing, politely ask the user for them in a short list."""


def route(state: AgentState) -> dict:
    raw = get_llm().invoke([("system", ROUTE_PROMPT), ("human", state["question"])]).content
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        intent = data.get("intent", "knowledge")
        params = data.get("params") or {}
    except (json.JSONDecodeError, AttributeError):
        intent, params = "knowledge", {}
    if intent not in {"crop", "weather", "knowledge", "chat"}:
        intent = "knowledge"
    return {"intent": intent, "params": params}


def crop_tool(state: AgentState) -> dict:
    return {"tool_result": recommend_crop(state["params"])}


def weather_tool(state: AgentState) -> dict:
    return {"tool_result": get_weather(state["params"].get("city", ""))}


def docs_tool(state: AgentState) -> dict:
    return {"tool_result": search_docs(state["question"])}


def chat_tool(state: AgentState) -> dict:
    return {"tool_result": {}}


def respond(state: AgentState) -> dict:
    if state["intent"] == "chat":
        messages = [
            ("system", "You are a friendly farming assistant. Reply briefly and invite a farming question."),
            ("human", state["question"]),
        ]
    else:
        messages = [
            ("system", ANSWER_PROMPT),
            ("human", f"Question: {state['question']}\n\nTool result:\n{json.dumps(state['tool_result'])}"),
        ]
    return {"answer": get_llm().invoke(messages).content}


builder = StateGraph(AgentState)
builder.add_node("route", route)
builder.add_node("crop_tool", crop_tool)
builder.add_node("weather_tool", weather_tool)
builder.add_node("docs_tool", docs_tool)
builder.add_node("chat_tool", chat_tool)
builder.add_node("respond", respond)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    lambda s: s["intent"],
    {"crop": "crop_tool", "weather": "weather_tool", "knowledge": "docs_tool", "chat": "chat_tool"},
)
for node in ("crop_tool", "weather_tool", "docs_tool", "chat_tool"):
    builder.add_edge(node, "respond")
builder.add_edge("respond", END)

graph = builder.compile()


def run_agent(question: str) -> dict:
    out = graph.invoke({"question": question})
    return {"answer": out["answer"], "intent": out["intent"], "tool_result": out.get("tool_result")}