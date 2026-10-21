import json
import re
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
MARKDOWN_DIR = DATASET_DIR / "EnterpriseRAG_2025_02_markdown"


# PAGE_MARKER - breakdown
# r'...': Denotes a "raw string" in Python. 
#   This ensures that backslashes (\) are treated as literal characters and not Python escape sequences.
# ^: Matches the beginning of a line (because of the re.MULTILINE flag).
# \{ and \}: Matches literal opening { and closing } curly braces. 
#   (The backslashes are required because curly braces normally define a quantity range in regex).
# (\d+): This is a capturing group that matches one or more digits (0-9). The code will remember this number, allowing you to extract the exact page number later.
# - +: Matches one or more literal hyphens/dashes.
# \s*: Matches zero or more trailing spaces or tabs.
# $: Matches the end of a line (because of the re.MULTILINE flag).

PAGE_MARKER = re.compile(r"^\{(\d+)\}-+\s*$", re.MULTILINE)

def load_companies() -> dict[str, str]:
    """Map each report's sha1 to its company name."""

    with open(DATASET_DIR / "subset.json", "r") as f:
        rows = json.load(f)
    return {row["sha1"]: row["company_name"] for row in rows}

def clean_text(text: str) -> str:
    """Remove page markers and extra whitespace from text."""
    # fixing 2 or more spaces to 1 space, but keeping newlines
    text = re.sub(r" {2,}", " ", text)
    # Shorten table separator rows: --------------- becomes ---
    text = re.sub(r"-{3,}", "---", text)
    # Replace HTML line breaks with a space-separated semicolon
    text = re.sub(r"<br\s*/?>", " ; ", text, flags=re.IGNORECASE)
    # Replace image links with an empty string (removing them)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>', '', text, flags=re.IGNORECASE)
    # Remove any remaining HTML tags
    text = re.sub(r'\\\$', '$', text)

    return text


def load_pages(sha1: str) -> list[dict]:
    """Split one report into pages: [{"page": 1, "text": "..."}, ...]."""
    md_path = MARKDOWN_DIR / sha1 / f"{sha1}.md"
    text = md_path.read_text(encoding="utf-8")

    # Because the regex has a capture group, re.split keeps the page numbers:
    # ["", "0", "<page 0 text>", "1", "<page 1 text>", ...]
    parts = PAGE_MARKER.split(text)

    pages = []
    for num, body in zip(parts[1::2], parts[2::2]):
        body = clean_text(body).strip()
        if body:  # skip empty pages
            # markers are 0-based, PDF pages are 1-based
            pages.append({"page": int(num)+1, "text": body})

    return pages

if __name__ == "__main__":
    companies = load_companies()
    company_pages = []
    all_pages = []

    for sha1, company in companies.items():
        pages = load_pages(sha1)
        company_pages.append((company, sha1, len(pages)))
        all_pages.extend(
            (company, page["page"], len(page["text"])) for page in pages
        )

    company_pages.sort(key=lambda x: x[2], reverse=True)
    all_pages.sort(key=lambda x: x[2], reverse=True)

    for company, sha1, page_count in company_pages:
        print(f"{company} ({sha1}) has {page_count} pages.")

    # longest pages
    for company, page_num, text_length in all_pages[:5]:
        print(f"{company} — Page {page_num}: {text_length:,} characters")