import re
import psycopg
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.retrieval import DB_URI, ensure_table, embed, to_pgvector

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def clean(text: str) -> str:
    text = text.replace("\x7f", "•")
    # header pattern specific to the rulebook PDF; harmless for other files
    text = re.sub(r"Release Management Rulebook \| General enterprise template\s*Page \d+", "", text)
    text = re.sub(r"_{3,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_pdf(path: str, source_name: str) -> int:
    """Chunk, embed and store a PDF. Returns the number of chunks stored."""
    ensure_table()
    chunks = []  # (page, chunk_index, text)
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = clean(page.extract_text() or "")
        for i, piece in enumerate(splitter.split_text(text)):
            if len(piece.strip()) >= 40:
                chunks.append((page_no, i, piece))

    if not chunks:
        return 0

    vectors = embed([c[2] for c in chunks])
    with psycopg.connect(DB_URI) as conn:
        conn.execute("DELETE FROM doc_chunks WHERE source = %s", (source_name,))  # re-ingest safe
        with conn.cursor() as cur:
            for (page_no, i, text), vec in zip(chunks, vectors):
                cur.execute(
                    "INSERT INTO doc_chunks (source, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector)",
                    (source_name, page_no, i, text, to_pgvector(vec)),
                )
    return len(chunks)