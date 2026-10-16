# MemoryMate

A memory-aware RAG assistant built with **LangGraph**, **PostgreSQL + pgvector**, **FastAPI** and **Streamlit**.

It answers questions from uploaded PDFs with citations, remembers conversations across restarts, and keeps long-term facts about each user across brand-new conversations. I built it to understand how an AI agent manages state and memory, and the demo domain is release management (it runs against a release management rulebook).

> This is a learning project, not a production system. See [Limitations](#limitations-and-roadmap).

## What it does

- **Answers from documents (RAG).** PDFs are chunked, embedded and stored in PostgreSQL with pgvector. Answers cite the file and page.
- **Says when it doesn't know.** If the documents don't contain the answer, it says so instead of inventing one.
- **Short-term memory.** A conversation is saved per thread and can be resumed after the app restarts.
- **Long-term memory.** Durable facts about a user (name, role, how they like explanations) are extracted, saved, updated in place, and reused in new threads. Users can delete them.
- **Routing.** A router node decides per message whether the documents need to be searched at all.
- **Three ways to use it:** a Streamlit chat UI, a FastAPI backend (with interactive docs), and a command-line chat.

## Architecture

```mermaid
flowchart TD
    UI[Streamlit UI] --> API[FastAPI]
    API --> LM[load_memory]
    LM --> RT{route_question}
    RT -- needs documents --> RE[retrieve]
    RT -- no --> GE[generate]
    RE --> GE
    GE --> EM[extract_memory]
    EM --> OUT[Answer + sources]
    PG[(PostgreSQL + pgvector)]
    LM -. read facts .-> PG
    RE -. vector search .-> PG
    EM -. write facts .-> PG
```

Each turn: restore the thread's history, load the user's facts, optionally search the documents, generate an answer, extract any new facts, then save the updated thread state.

### Two kinds of memory

| | Short-term memory | Long-term memory |
|---|---|---|
| Mechanism | `PostgresSaver` (LangGraph checkpointer) | `PostgresStore` |
| Keyed by | `thread_id` | `user_id` (namespace `users/<id>/facts`) |
| Holds | Messages and graph state of one conversation | Short facts about the user, one sentence per key |
| Written by | LangGraph, automatically after every step | My own `extract_memory` node |
| Survives | Restarts, but only helps inside that thread | Restarts and new threads |

The knowledge base lives in a separate `doc_chunks` table: 384-dimension vectors (all-MiniLM-L6-v2 via FastEmbed) with an HNSW cosine index.

## Tech stack

- **Orchestration:** LangGraph, LangChain (Groq chat model)
- **Database:** PostgreSQL, pgvector, `langgraph-checkpoint-postgres`, psycopg 3
- **Embeddings:** FastEmbed (ONNX, no PyTorch needed)
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit
- **PDF and chunking:** pypdf, langchain-text-splitters

## Project structure

```
app/
├── graph.py        # LangGraph workflow: memory, router, retrieval, generation
├── retrieval.py    # embeddings, doc_chunks table, vector search
├── ingestion.py    # PDF -> clean -> chunk -> embed -> store
└── api.py          # FastAPI endpoints
ui/
└── streamlit_app.py
scripts/
├── chat_cli.py     # command-line chat
├── ingest.py       # ingest a PDF from the terminal
├── memories.py     # list / delete / clear a user's stored facts
└── search_demo.py  # inspect retrieval scores for a question
```

## Getting started

### Prerequisites

- Python 3.10 or newer
- PostgreSQL 16 or newer with the `pgvector` extension
- A Groq API key

### 1. Install

```bash
git clone <your-repo-url>
cd memorymate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up PostgreSQL

Using a local install:

```sql
CREATE USER memorymate WITH PASSWORD 'your_password';
CREATE DATABASE memorymate OWNER memorymate;
\c memorymate
CREATE EXTENSION IF NOT EXISTS vector;
```

Or with Docker:

```bash
docker run -d --name memorymate-db \
  -e POSTGRES_USER=memorymate -e POSTGRES_PASSWORD=your_password -e POSTGRES_DB=memorymate \
  -p 5432:5432 pgvector/pgvector:pg16
```

The app creates its own tables (checkpoints, store, `doc_chunks`) on first start.

### 3. Configure

Create a `.env` file (never commit it):

```
GROQ_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-20b
DB_URI=postgresql://memorymate:your_password@localhost:5432/memorymate
```

Model names change over time and depend on your Groq account. List the models your key can use with the Groq API and pick a chat model.

### 4. Add documents

Put a text-based PDF in `data/` and ingest it (scanned PDFs need OCR, which is not included):

```bash
python -m scripts.ingest data/your_document.pdf
```

You can also upload PDFs from the Streamlit sidebar.

### 5. Run

In one terminal, the API:

```bash
uvicorn app.api:app --port 8000
```

In another, the UI:

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 for the chat UI and http://localhost:8000/docs for the API docs.

Prefer the terminal? `python -m scripts.chat_cli <thread_id> <user_id>`

## Try it

1. Ask a question your PDF answers. The answer cites the page, and the "Sources used" panel shows the retrieved chunks and scores.
2. Ask something the document doesn't cover. It should say the documents don't contain enough evidence.
3. Say "My name is Sam and I prefer short bullet-point answers." New facts appear in the sidebar.
4. Click **New thread** and ask "What is my name?". The answer comes from long-term memory, not from the thread.
5. Restart both services, enter the old thread ID and click **Load thread**. The conversation is restored from PostgreSQL.
6. Delete a fact in the sidebar and ask again in a new thread. It no longer knows it.

## Design notes

- **Memory scoping lives in application code.** The `user_id` comes from the request config, never from the model, and thread IDs are stored as `user__thread` so users can't open each other's threads.
- **Per-turn state is reset.** The state is checkpointed, so `docs` and `searched` are cleared at the start of every turn. Otherwise last turn's excerpts leak into the next answer.
- **Stable keys turn duplicates into updates.** The extraction step sees the existing facts with their keys and reuses a key to update a fact, so "I now prefer bullets" replaces the old preference instead of adding a second one.
- **A score threshold only catches off-topic questions.** A question like "what dollar amount needs CAB approval?" scores about as high as a real match because the document talks about approvals, even though the amount isn't there. The prompt ("use only the excerpts, say so if the answer isn't there") handles that case.
- **Cleaning the PDF text mattered.** Removing bullet control characters, repeated page headers and blank fill-in lines raised the top similarity score for a test question from 0.42 to 0.56.
- **Memory extraction is best-effort.** If it fails, the chat carries on. Losing one fact is better than breaking a conversation.

## Limitations and roadmap

Known limitations:

- **No authentication.** The user ID is whatever the client sends, so this demo is not safe for multiple real users. A real deployment would derive it from a verified token.
- **One shared document library.** All users search the same `doc_chunks` table.
- **Retrieval uses only the latest message**, so follow-ups like "who attends it?" retrieve poorly.
- **Table-heavy PDFs lose some cell text** during extraction.
- **Single database connections, no pooling, no streaming responses.**
- **No automated evaluation** of retrieval or answer quality yet.

Planned:

- Query rewriting that uses conversation history before routing and searching
- Semantic search over memories instead of loading all of them
- Per-user documents and real authentication
- Human-in-the-loop approval for release actions (LangGraph interrupts)
- A small evaluation set for retrieval and grounding

## License

Add a license of your choice (for example MIT).