import sys
from app.retrieval import search_chunks

# Usage: python -m scripts.search_demo "your question"
query = sys.argv[1]
results = search_chunks(query, k=5, min_score=0.0)  # min_score=0 shows everything

print(f"Q: {query}\n")
for r in results:
    print(f"{r['score']:.2f}  [{r['source']} p.{r['page']}]  {r['content'][:120]!r}\n")