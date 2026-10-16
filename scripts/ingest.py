import os
import sys
from app.ingestion import ingest_pdf

# Usage: python -m scripts.ingest data/file.pdf
path = sys.argv[1]
name = os.path.basename(path)
print(f"{ingest_pdf(path, name)} chunks ingested from {name}")