import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.environ["DB_URI"]

with psycopg.connect(DB_URI, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        print("Connected:", cur.fetchone()[0])

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
        print("pgvector:", cur.fetchone())

        # Persistence test table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS connection_test (
                id SERIAL PRIMARY KEY,
                note TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        cur.execute("INSERT INTO connection_test (note) VALUES ('hello from phase 1');")
        cur.execute("SELECT count(*) FROM connection_test;")
        print("Rows in connection_test:", cur.fetchone()[0])