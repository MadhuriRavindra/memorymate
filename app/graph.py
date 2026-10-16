import os
import re
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from app.retrieval import search_chunks

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
llm = ChatGroq(model=MODEL, temperature=0.3)
extractor = ChatGroq(model=MODEL, temperature=0)  # deterministic: router + memory extraction

BASE_PROMPT = (
    "You are MemoryMate, a helpful assistant for release management and AI "
    "engineering questions. Be clear and concise."
)

MAX_FACTS_PER_TURN = 5
MAX_FACT_LENGTH = 200


class State(MessagesState):
    memories: list[dict]   # [{"key": ..., "fact": ...}] loaded for this turn
    docs: list[dict]       # document chunks retrieved this turn
    searched: bool         # did we search the documents this turn?


def user_ns(config):
    # user_id comes from the application config, never from the LLM
    return ("users", config["configurable"]["user_id"], "facts")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40]


# ---------- nodes ----------

def load_memory(state: State, config, *, store: BaseStore):
    items = store.search(user_ns(config), limit=50)
    # Reset per-turn fields: state is checkpointed, so old docs would otherwise leak in.
    return {
        "memories": [{"key": i.key, "fact": i.value["fact"]} for i in items],
        "docs": [],
        "searched": False,
    }


def route_question(state: State) -> str:
    question = state["messages"][-1].content
    prompt = (
        "Decide if answering the message below needs information from the user's "
        "uploaded documents (policies, runbooks, PDFs, domain content).\n"
        "Answer YES for questions about document content or specific procedures.\n"
        "Answer NO for greetings, small talk, questions about the user themselves "
        "(their name, preferences), or general knowledge questions.\n"
        "Reply with only YES or NO.\n\n"
        f"Message: {question}"
    )
    try:
        answer = extractor.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        decision = "retrieve" if answer.startswith("YES") else "generate"
    except Exception as e:
        print("[route] failed:", e)
        decision = "retrieve"  # safe default
    print(f"[route] {decision}")
    return decision


def retrieve(state: State):
    docs = search_chunks(state["messages"][-1].content)
    print("[retrieve]", [(d["page"], round(d["score"], 2)) for d in docs])
    return {"docs": docs, "searched": True}


def generate(state: State):
    parts = [BASE_PROMPT]

    if state.get("memories"):
        parts.append(
            "Known facts about the user (use them to personalise answers):\n- "
            + "\n- ".join(m["fact"] for m in state["memories"])
        )

    if state.get("docs"):
        context = "\n\n".join(
            f"[{d['source']} p.{d['page']}] {d['content']}" for d in state["docs"]
        )
        parts.append(
            "Answer using ONLY the document excerpts below and cite them like "
            "[file p.N]. If they don't contain the answer, say so.\n\n" + context
        )
    elif state.get("searched"):
        parts.append(
            "The user's documents were searched and nothing relevant was found. "
            "Say the documents don't contain enough evidence to answer. "
            "Do not invent an answer."
        )

    response = llm.invoke([SystemMessage(content="\n\n".join(parts))] + state["messages"])

    if not str(response.content).strip():
        print("[debug] EMPTY reply. metadata:", response.response_metadata,
              "| extra keys:", list(response.additional_kwargs))
    return {"messages": [response]}


def extract_memory(state: State, config, *, store: BaseStore):
    last_user = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if not last_user:
        return {}

    existing = {m["key"]: m["fact"] for m in state.get("memories", [])}
    existing_text = "\n".join(f"{k}: {v}" for k, v in existing.items()) or "(none)"

    prompt = (
        "You maintain a long-term memory of durable facts about the USER "
        "(name, role, goals, preferences for how they like explanations).\n\n"
        f"Current memory (key: fact):\n{existing_text}\n\n"
        f"New user message:\n{last_user}\n\n"
        "Return ONLY a JSON array of objects like "
        '{"key": "explanation_style", "fact": "User prefers short bullet-point explanations."}.\n'
        "Rules:\n"
        "- If the message corrects or changes an existing fact, REUSE its key so it is updated.\n"
        "- For a genuinely new fact, use a new short snake_case key (e.g. name, role, goal).\n"
        "- Each fact must be a complete, self-contained sentence.\n"
        "- Only save facts about the user themselves, not questions about documents.\n"
        "- Do NOT save passwords, keys, financial, health or other sensitive data.\n"
        "- Do NOT save facts already in memory unchanged.\n"
        "- If there is nothing worth saving, return []."
    )
    try:
        raw = extractor.invoke([HumanMessage(content=prompt)]).content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        updates = json.loads(raw)
    except Exception as e:
        print("[memory] extraction failed:", e)
        return {}

    if not isinstance(updates, list):
        return {}

    for u in updates[:MAX_FACTS_PER_TURN]:
        if not isinstance(u, dict):
            continue
        key, fact = slug(u.get("key", "")), str(u.get("fact", "")).strip()
        if not key or not fact or len(fact) > MAX_FACT_LENGTH:
            continue
        if existing.get(key) == fact:
            continue
        store.put(user_ns(config), key, {"fact": fact})
    return {}


# ---------- graph ----------

builder = StateGraph(State)
builder.add_node("load_memory", load_memory)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("extract_memory", extract_memory)

builder.add_edge(START, "load_memory")
builder.add_conditional_edges("load_memory", route_question, ["retrieve", "generate"])
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "extract_memory")
builder.add_edge("extract_memory", END)


def build_graph(checkpointer=None, store=None):
    return builder.compile(checkpointer=checkpointer, store=store)