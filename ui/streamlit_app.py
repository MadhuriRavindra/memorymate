import os
import uuid

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MemoryMate", page_icon="🧠", layout="wide")


def call(method, path, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}: {e}")
        return None
    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text[:300]}")
        return None
    return r.json()


# ---------- session state ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"t-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "madhuri"


def new_thread():
    st.session_state.thread_id = f"t-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []


def load_thread():
    data = call("GET", "/history", params={
        "user_id": st.session_state.user_id,
        "thread_id": st.session_state.thread_id,
    })
    st.session_state.messages = data or []


# ---------- sidebar ----------
with st.sidebar:
    st.header("Session")
    st.text_input("User ID", key="user_id")
    st.text_input("Thread ID", key="thread_id")
    c1, c2 = st.columns(2)
    c1.button("New thread", on_click=new_thread, use_container_width=True)
    c2.button("Load thread", on_click=load_thread, use_container_width=True)
    st.caption("Type an existing thread ID and press Load to resume it.")

    st.header("Documents")
    for d in call("GET", "/documents") or []:
        st.write(f"📄 {d['source']} ({d['chunks']} chunks)")
    up = st.file_uploader("Upload a PDF", type="pdf")
    if up and st.button("Ingest PDF"):
        with st.spinner("Chunking and embedding..."):
            res = call("POST", "/documents",
                       files={"file": (up.name, up.getvalue(), "application/pdf")})
        if res:
            st.success(f"{res['source']}: {res['chunks']} chunks stored")
            st.rerun()

    st.header("Long-term memory")
    memories = call("GET", f"/memories/{st.session_state.user_id}") or []
    if not memories:
        st.caption("Nothing remembered yet.")
    for m in memories:
        col_a, col_b = st.columns([5, 1])
        col_a.write(f"• {m['fact']}")
        if col_b.button("✕", key=f"del_{m['key']}"):
            call("DELETE", f"/memories/{st.session_state.user_id}/{m['key']}")
            st.rerun()

# ---------- main chat ----------
st.title("🧠 MemoryMate")
st.caption("LangGraph · PostgreSQL · pgvector · short-term + long-term memory + RAG")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("searched"):
            with st.expander("Sources used"):
                if m["sources"]:
                    for s in m["sources"]:
                        st.write(f"{s['source']} p.{s['page']} (score {s['score']})")
                else:
                    st.write("Documents searched, nothing relevant found.")
        if m.get("memories_used"):
            with st.expander("Memories used"):
                for f in m["memories_used"]:
                    st.write(f"• {f}")

if prompt := st.chat_input("Ask about the rulebook, or tell me about yourself..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = call("POST", "/chat", json={
                "user_id": st.session_state.user_id,
                "thread_id": st.session_state.thread_id,
                "message": prompt,
            })
    if data:
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "searched": data["searched_documents"],
            "sources": data["sources"],
            "memories_used": data["memories_used"],
        })
        st.rerun()  # refresh the sidebar so new memories appear