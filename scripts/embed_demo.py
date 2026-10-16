import os
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "A release freeze stops new code changes before a production deployment.",
    "A rollback restores the previous version when a deployment fails.",
    "The change advisory board approves high-risk changes.",
    "Python is a popular programming language for data and AI.",
    "Lentils and vegetables make a healthy vegetarian curry.",
]

queries = [
    "What do we do if a deployment goes wrong?",
    "Who signs off on risky changes?",
    "Tell me about cooking.",
]


def to_vec(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


vectors = model.encode(sentences, normalize_embeddings=True).tolist()
print("Vector length:", len(vectors[0]))
print("First 5 numbers of sentence 1:", [round(x, 3) for x in vectors[0][:5]], "\n")

with psycopg.connect(os.environ["DB_URI"], autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("DROP TABLE IF EXISTS embedding_demo")
    conn.execute(
        "CREATE TABLE embedding_demo ("
        "id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))"
    )
    for text, vec in zip(sentences, vectors):
        conn.execute(
            "INSERT INTO embedding_demo (content, embedding) VALUES (%s, %s::vector)",
            (text, to_vec(vec)),
        )

    for q in queries:
        qv = to_vec(model.encode([q], normalize_embeddings=True)[0].tolist())
        rows = conn.execute(
            "SELECT content, 1 - (embedding <=> %s::vector) AS score "
            "FROM embedding_demo ORDER BY embedding <=> %s::vector LIMIT 3",
            (qv, qv),
        ).fetchall()
        print(f"Q: {q}")
        for content, score in rows:
            print(f"   {score:.2f}  {content}")
        print()