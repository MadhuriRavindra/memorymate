import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from app.graph import build_graph
from app.retrieval import ensure_table

load_dotenv()
DB_URI = os.environ["DB_URI"]

# Usage: python -m scripts.chat_cli [thread_id] [user_id]
thread_id = sys.argv[1] if len(sys.argv) > 1 else "demo-1"
user_id = sys.argv[2] if len(sys.argv) > 2 else "madhuri"
config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

ensure_table()

with PostgresSaver.from_conn_string(DB_URI) as checkpointer, \
     PostgresStore.from_conn_string(DB_URI) as store:
    checkpointer.setup()
    store.setup()
    graph = build_graph(checkpointer, store)

    print(f"MemoryMate - thread: {thread_id} | user: {user_id} (type 'exit' to quit)")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        print("Bot:", result["messages"][-1].content)