"""Step 4: find the chunks most likely to answer a question."""
import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from rag.embed import INDEX_DIR, embed_query
from rag.ingest import load_companies

TOP_K = 5
CANDIDATES = 20  # each retriever proposes this many before fusion
RRF_K = 60       # standard constant from the Reciprocal Rank Fusion paper

# Instructions aimed at the answer format, not at the content we're searching for
BOILERPLATE = re.compile(r"\s*If (?:there is no mention|data is not available).*$", re.IGNORECASE | re.DOTALL)


def find_companies(question: str) -> list[tuple[str, str]]:
    """Route a question to the reports it names: [(sha1, company), ...]."""
    return [(sha1, name) for sha1, name in load_companies().items() if name in question]


def clean_query(question: str, company: str) -> str:
    """Drop the company name (every chunk in this index shares it) and answer-format boilerplate."""
    return BOILERPLATE.sub("", question.replace(company, "")).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=None)
def load_index(sha1: str) -> tuple[np.ndarray, list[dict], BM25Okapi]:
    """Load one company's vectors + chunks and build its BM25 index (cached per process)."""
    vectors = np.load(INDEX_DIR / f"{sha1}.npy")
    chunks = json.loads((INDEX_DIR / f"{sha1}.json").read_text(encoding="utf-8"))
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return vectors, chunks, bm25


def vector_search(query: str, vectors: np.ndarray, k: int) -> list[int]:
    scores = vectors @ embed_query(query)
    return np.argsort(-scores)[:k].tolist()


def bm25_search(query: str, bm25: BM25Okapi, k: int) -> list[int]:
    scores = bm25.get_scores(tokenize(query))
    return np.argsort(-scores)[:k].tolist()


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) across rankings."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(question: str, sha1: str, company: str, mode: str = "hybrid", top_k: int = TOP_K) -> list[dict]:
    vectors, chunks, bm25 = load_index(sha1)
    query = clean_query(question, company)
    if mode == "vector":
        ids = vector_search(query, vectors, top_k)
    elif mode == "bm25":
        ids = bm25_search(query, bm25, top_k)
    else:
        ids = rrf([vector_search(query, vectors, CANDIDATES), bm25_search(query, bm25, CANDIDATES)])[:top_k]
    return [chunks[i] for i in ids]


if __name__ == "__main__":
    questions = [
        "According to the annual report, what is the Cash flow from operations (in USD) for Sonic Automotive, Inc.  (within the last period or at the end of the last period)? If data is not available, return 'N/A'.",
        "Did Liberty Broadband Corporation announce a share buyback plan in the annual report? If there is no mention, return False.",
        "How many bank locations does Guaranty Bancshares, Inc. have?",
    ]
    for question in questions:
        (sha1, company), = find_companies(question)
        print(f"\nQ: {clean_query(question, company)}   [{company}]")
        for mode in ("vector", "bm25", "hybrid"):
            hits = retrieve(question, sha1, company, mode)
            print(f"  {mode:<7} pages {[c['page'] for c in hits]}  top: {hits[0]['text'][:70]!r}")
