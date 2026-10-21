import json
import time
from pathlib import Path

import numpy as np
import requests

from rag.chunking import build_chunks
from rag.ingest import load_companies, load_pages

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 64  # measured: ~75 chunks/s at 64 vs ~64 chunks/s at 16
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"

# nomic-embed-text was trained with task prefixes; documents and queries get different ones
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts in batches. Returns float32 array of shape (len(texts), 768)."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "input": texts[i:i + BATCH_SIZE]},
            timeout = 300,
        )
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    return np.array(vectors, dtype=np.float32)

def embed_query(question: str) -> np.ndarray:
    return embed_texts([QUERY_PREFIX + question])[0]

def document_text(chunk: dict) -> str:
    """Reconstruct the text of a chunk for embedding, with the task prefix."""
    return f"{DOC_PREFIX}{chunk['company']}\n{chunk['text']}"

def index_company(sha1: str, company: str) -> int:
    """Embed one report and save vectors + chunk metadata. Skips if already cached."""
    vec_path = INDEX_DIR / f"{sha1}.npy"
    meta_path = INDEX_DIR / f"{sha1}.json"
    if vec_path.exists() and meta_path.exists():
        return 0

    chunks = build_chunks(sha1, company)
    vectors = embed_texts([document_text(c) for c in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    meta_path.write_text(json.dumps(chunks), encoding="utf-8")
    np.save(vec_path, vectors)  # written last: its presence means this company is complete
    return len(chunks)

def build_index() -> None:
    companies = load_companies()
    start = time.time()
    for n, (sha1, company) in enumerate(companies.items(), 1):
        count = index_company(sha1, company)
        status = f"embedded {count:,} chunks" if count else "cached"
        print(f"[{n:>3}/{len(companies)}] {company}: {status} ({time.time() - start:.0f}s)")

if __name__ == "__main__":
    build_index()