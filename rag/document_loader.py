import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge" / "raw"


def clean_text(text: str) -> str:
    """Basic cleanup while preserving line structure."""

    replacements = {
        "â€™": "'",
        "â€“": "-",
        "â€”": "-",
        "Ïƒ": "σ",
        "\x00": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines = []

    for line in text.splitlines():
        # 여러 공백과 탭을 하나의 공백으로 정리
        line = re.sub(r"[ \t]+", " ", line).strip()

        # 빈 줄은 제외
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def load_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)

        pages.append(
            {
                "source": pdf_path.name,
                "page": page_number,
                "text": text,
                "char_count": len(text),
            }
        )

    return pages


def load_all_documents() -> list[dict]:
    documents = []

    pdf_files = sorted(KNOWLEDGE_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        documents.extend(pages)

    return documents


if __name__ == "__main__":
    documents = load_all_documents()

    print(f"Loaded pages: {len(documents)}")

    for doc in documents:
        status = "OK" if doc["char_count"] >= 50 else "CHECK"

        print(
            f"[{status}] "
            f"{doc['source']} "
            f"page={doc['page']} "
            f"chars={doc['char_count']}"
        )