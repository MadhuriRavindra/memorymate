import os
from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore

load_dotenv()

with PostgresStore.from_conn_string(os.environ["DB_URI"]) as store:
    store.setup()  # creates the store table

    ns = ("users", "madhuri", "facts")
    store.put(ns, "fact-1", {"fact": "Prefers simple Python explanations"})
    store.put(ns, "fact-2", {"fact": "Preparing for AI developer interviews"})

    print("get:", store.get(ns, "fact-1").value)

    print("search madhuri:")
    for item in store.search(ns):
        print("  ", item.key, item.value)

    print("search someone-else:", store.search(("users", "someone-else", "facts")))