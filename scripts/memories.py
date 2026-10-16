import os
import sys
from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore

load_dotenv()

# Usage:
#   python -m scripts.memories madhuri             -> list facts
#   python -m scripts.memories madhuri delete 2    -> delete fact number 2
#   python -m scripts.memories madhuri clear       -> delete all facts for the user
user_id = sys.argv[1] if len(sys.argv) > 1 else "madhuri"
ns = ("users", user_id, "facts")

with PostgresStore.from_conn_string(os.environ["DB_URI"]) as store:
    items = store.search(ns, limit=100)

    if len(sys.argv) >= 3 and sys.argv[2] == "clear":
        for it in items:
            store.delete(ns, it.key)
        print(f"Cleared {len(items)} facts.\n")
        items = []
    elif len(sys.argv) >= 4 and sys.argv[2] == "delete":
        idx = int(sys.argv[3]) - 1
        store.delete(ns, items[idx].key)
        print(f"Deleted: {items[idx].value['fact']}\n")
        items = store.search(ns, limit=100)

    print(f"Facts for user '{user_id}':")
    for n, item in enumerate(items, start=1):
        print(f"  {n}. [{item.key}] {item.value['fact']}")