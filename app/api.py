import os
import re
import tempfile
from contextlib import asynccontextmanager

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field

from app.graph import build_graph
from app.ingestion import ingest_pdf
from app.retrieval import DB_URI, ensure_table, get_model

load_dotenv()

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
ID_RE = re.compile(ID_PATTERN)
MAX_UPLOAD_MB = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_table()
    get_model()  # load the embedding model now so the first request isn't slow
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer, \
         PostgresStore.from_conn_string(DB_URI) as store:
        checkpointer.setup()
        store.setup()
        app.state.store = store
        app.state.graph = build_graph(checkpointer, store)
        yield


app = FastAPI(title="MemoryMate API", lifespan=lifespan)


# ---------- helpers ----------

def check_id(value: str, name: str):
    if not ID_RE.match(value):
        raise HTTPException(422, f"Invalid {name}: use letters, digits, - or _ (max 64)")


def make_config(user_id: str, thread_id: str) -> dict:
    # Threads are keyed per user, so one user can't open another user's thread.
    return {"configurable": {"thread_id": f"{user_id}__{thread_id}", "user_id": user_id}}


def facts_ns(user_id: str):
    return ("users", user_id, "facts")


# ---------- chat ----------

class ChatRequest(BaseModel):
    user_id: str = Field(pattern=ID_PATTERN)
    thread_id: str = Field(pattern=ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


@app.post("/chat")
def chat(req: ChatRequest):
    config = make_config(req.user_id, req.thread_id)
    try:
        result = app.state.graph.invoke(
            {"messages": [HumanMessage(content=req.message)]}, config
        )
    except Exception as e:
        raise HTTPException(502, f"Graph error: {e}")

    return {
        "answer": result["messages"][-1].content,
        "searched_documents": bool(result.get("searched")),
        "sources": [
            {"source": d["source"], "page": d["page"], "score": round(d["score"], 2)}
            for d in result.get("docs", [])
        ],
        "memories_used": [m["fact"] for m in result.get("memories", [])],
    }


@app.get("/history")
def history(user_id: str, thread_id: str):
    check_id(user_id, "user_id")
    check_id(thread_id, "thread_id")
    state = app.state.graph.get_state(make_config(user_id, thread_id))
    messages = (state.values or {}).get("messages", []) if state else []
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in messages
        if m.content
    ]


# ---------- long-term memory ----------

@app.get("/memories/{user_id}")
def list_memories(user_id: str):
    check_id(user_id, "user_id")
    items = app.state.store.search(facts_ns(user_id), limit=100)
    return [{"key": i.key, "fact": i.value["fact"]} for i in items]


@app.delete("/memories/{user_id}/{key}")
def delete_memory(user_id: str, key: str):
    check_id(user_id, "user_id")
    app.state.store.delete(facts_ns(user_id), key)
    return {"deleted": key}


# ---------- documents ----------

@app.get("/documents")
def list_documents():
    with psycopg.connect(DB_URI) as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) FROM doc_chunks GROUP BY source ORDER BY source"
        ).fetchall()
    return [{"source": r[0], "chunks": r[1]} for r in rows]


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        count = ingest_pdf(path, source_name=name)
    finally:
        os.remove(path)

    if count == 0:
        raise HTTPException(422, "No text could be extracted (is it a scanned PDF?)")
    return {"source": name, "chunks": count}