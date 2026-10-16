import os
import numpy as np
import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
DB_URI = os.environ["DB_URI"]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # must match the model; if you change models, recreate the table

_model = None


def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    vecs = np.array(list(get_model().embed(texts)))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # normalise
    return vecs.tolist()


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def ensure_table():
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                page INT,
                chunk_index INT,
                content TEXT NOT NULL,
                embedding vector({EMBED_DIM})
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
            ON doc_chunks USING hnsw (embedding vector_cosine_ops);
        """)


def search_chunks(query: str, k: int = 5, min_score: float = 0.2) -> list[dict]:
    qvec = to_pgvector(embed([query])[0])
    with psycopg.connect(DB_URI) as conn:
        rows = conn.execute(
            """
            SELECT content, source, page, 1 - (embedding <=> %s::vector) AS score
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, k),
        ).fetchall()
    results = [
        {"content": r[0], "source": r[1], "page": r[2], "score": float(r[3])}
        for r in rows
    ]
    return [r for r in results if r["score"] >= min_score]