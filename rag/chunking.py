"""Step 2b: split cleaned pages into retrieval-sized chunks."""

from rag.ingest import load_companies, load_pages

MAX_CHARS = 1600     # ~350 tokens of prose, ~710 worst case for tables: safely under the 2048 limit
OVERLAP_CHARS = 200  # ~12% of MAX_CHARS
SEPARATORS = ["\n\n", "\n", " "]  # paragraph -> line / table row -> word

def split_text(text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS, separators: list[str] = SEPARATORS) -> list[str]:
    """Split text into chunks of at most `max_chars`, with `overlap_chars` overlap."""
    if len(text) <= max_chars:
        return [text]

    if not separators:  # no natural boundary left: hard cut
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    sep, finder = separators[0], separators[1:]
    chunks, current = [], ""

    for piece in text.split(sep):
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate  # greedy packing: keep adding while it fits
            continue
        if current:  # flush the current chunk
            chunks.append(current)
        if len(piece) > max_chars:  # piece is too big: recurse with next separator
            chunks.extend(split_text(piece, max_chars, overlap_chars, finder))
            current = ""
        else:
            current = piece  # start a new chunk with the current piece

    if current:  # flush any remaining text
        chunks.append(current)

    return chunks

def add_overlap(chunks: list[str], overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
        """Prefix each chunk with the tail of the previous one, starting at a word boundary."""
        result = chunks[:1]  # first chunk has no overlap

        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap_chars:]
            if len(prev) > overlap_chars and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            result.append(f"{tail}\n{cur}")
        
        return result  # trim to the first space after the overlap

def build_chunks(sha1: str, company: str) -> list[dict]:
    """Chunk one report; every chunk keeps the metadata needed for filtering and citations."""
    chunks = []
    for page in load_pages(sha1):
        pieces = split_text(page["text"], MAX_CHARS - OVERLAP_CHARS)
        for index, text in enumerate(add_overlap(pieces, OVERLAP_CHARS)):
            chunks.append({
                "company": company,
                "sha1": sha1,
                "page": page["page"],
                "chunk": index + 1,
                "text": text.strip(),
            })
    return chunks
