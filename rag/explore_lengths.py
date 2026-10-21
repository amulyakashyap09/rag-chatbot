"""Step 2a: measure text lengths before choosing a chunk size."""
import numpy as np

# module.function_name to reuse functions from rag/ingest.py without circular imports
# TO run this file: python -m rag.explore_lengths
from rag.ingest import load_companies, load_pages


CHARS_PER_TOKEN = 4  # rough rule of thumb for English text
EMBED_MAX_TOKENS = 2048  # nomic-embed-text context length (`ollama show nomic-embed-text`)


def approx_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def summarize(name: str, lengths: list[int]) -> None:
    arr = np.array(lengths)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99]).astype(int)
    print(
        f"{name:<11} n={len(arr):>7,}  p50={p50:>5,}  p90={p90:>5,}  "
        f"p99={p99:>6,}  max={arr.max():>6,}  (approx tokens)"
    )


def main() -> None:
    page_lens, para_lens = [], []
    pages_with_table = 0

    for sha1 in load_companies():
        for page in load_pages(sha1):
            text = page["text"]
            page_lens.append(approx_tokens(text))
            para_lens.extend(approx_tokens(p) for p in text.split("\n\n") if p.strip())
            if any(line.lstrip().startswith("|") for line in text.splitlines()):
                pages_with_table += 1

    summarize("pages", page_lens)
    summarize("paragraphs", para_lens)

    too_long = sum(n > EMBED_MAX_TOKENS for n in para_lens)
    print(f"\nparagraphs over {EMBED_MAX_TOKENS} tokens: {too_long:,}")
    print(
        f"pages containing a table: {pages_with_table:,} / {len(page_lens):,} "
        f"({pages_with_table / len(page_lens):.0%})"
    )


if __name__ == "__main__":
    main()
