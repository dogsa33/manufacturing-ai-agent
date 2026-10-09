import re

from rag.document_loader import load_all_documents


TARGET_CHARS = 3000
MIN_PAGE_CHARS = 50
MIN_CHUNK_CHARS = 200
OVERLAP_LINES = 2


# 반복적으로 등장하지만 검색 지식으로는 가치가 낮은 문구
NOISE_LINES = {
    "READ POLICY",
    "SHOW LESS",
    "THE PROJECT",
    "NAVIGATION",
    "LOGISTICS",
    "Home",
    "View Datasets",
    "Donate a Dataset",
    "Contact",
}


def is_noise_line(line: str) -> bool:
    line = line.strip()

    if not line:
        return True

    upper = line.upper()
    lower = line.lower()

    # -------------------------
    # UCI web page noise
    # -------------------------
    uci_noise = {
        "READ POLICY",
        "ACCEPT READ POLICY",
        "SHOW LESS",
        "THE PROJECT",
        "NAVIGATION",
        "LOGISTICS",
        "HOME",
        "VIEW DATASETS",
        "DONATE A DATASET",
        "CONTACT",
        "DOWNLOAD (509.9 KB)",
        "IMPORT IN PYTHON",
        "CITE",
        "ABOUT US",
    }

    if upper in uci_noise:
        return True

    if "acknowledge and accept the cookies" in lower:
        return True

    if "practices used by the uci machine learning repository" in lower:
        return True

    if lower.startswith(
        "by using the uci machine learning repository"
    ):
        return True

    if re.fullmatch(r"\d+\s+views?", lower):
        return True

    if re.fullmatch(r"\d+\s+citations?", lower):
        return True

    # -------------------------
    # NASA recurring headers
    # -------------------------
    nasa_noise = {
        "NASA RELIABILITY-CENTERED MAINTENANCE GUIDE",
        "FOR FACILITIES AND COLLATERAL EQUIPMENT",
        "SEPTEMBER 2008",
        "DRAFT",
        "FINAL",
    }

    if upper in nasa_noise:
        return True

    # NASA page number: 1-4, 2-1, 6-23 ...
    if re.fullmatch(r"\d{1,2}-\d{1,3}", line):
        return True

    # -------------------------
    # DOE recurring footer/header
    # -------------------------
    if upper.startswith("O&M BEST PRACTICES GUIDE"):
        return True

    # -------------------------
    # NIST recurring footer
    # -------------------------
    if lower.startswith(
        "this publication is available free of charge from:"
    ):
        return True

    # -------------------------
    # Common page noise
    # -------------------------

    # 단독 숫자
    if re.fullmatch(r"\d+", line):
        return True

    # 단독 roman numeral: i, ii, iii ...
    if re.fullmatch(r"[ivxlcdm]+", lower):
        return True

    return False


def is_toc_page(text: str) -> bool:
    """
    목차 페이지를 검색 corpus에서 제외한다.
    """

    upper_text = text.upper()

    if "TABLE OF CONTENTS" in upper_text:
        return True

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    # "6.3 Vibration ........ 6-3" 같은 목차 형태
    dotted_lines = sum(
        1
        for line in lines
        if re.search(r"\.{5,}", line)
    )

    # 페이지의 상당 부분이 dot leader이면 목차로 판단
    return (
        dotted_lines >= 5
        and dotted_lines / len(lines) >= 0.25
    )


def is_heading(line: str) -> bool:
    """
    Conservative heading detection.
    False headings are worse than missing headings.
    """

    line = line.strip()

    if len(line) < 3 or len(line) > 120:
        return False

    # --------------------------------
    # 1. Numbered subsection
    # 2.1 PREVENTIVE MAINTENANCE
    # 6.3.2 Motor Analysis Test
    # --------------------------------
    if re.match(
        r"^\d+\.\d+(?:\.\d+){0,3}\s+\S+",
        line
    ):
        return True

    # --------------------------------
    # 2. Top-level numbered heading
    # NIST: 1 Introduction
    # NASA: 3 RCM Approach
    #
    # Citation과 혼동하지 않도록
    # 길이와 punctuation 제한
    # --------------------------------
    if re.match(r"^\d{1,2}\s+[A-Za-z]", line):

        if len(line) > 80:
            return False

        if "," in line or ";" in line:
            return False

        return True

    # --------------------------------
    # 3. Explicit Chapter
    # --------------------------------
    if re.match(
        r"^CHAPTER\s+\d+",
        line,
        re.IGNORECASE
    ):
        return True

    # --------------------------------
    # 4. Short question headings
    # OSHA:
    # Why is controlling hazardous energy important?
    # --------------------------------
    if (
        line.endswith("?")
        and len(line) <= 100
    ):
        return True

    # --------------------------------
    # 5. Known UCI headings
    # --------------------------------
    uci_headings = {
        "Dataset Information",
        "Additional Information",
        "Variables Table",
        "Additional Variable Information",
        "Dataset Files",
        "License",
        "Introductory Paper",
    }

    if line in uci_headings:
        return True

    return False


def clean_page_lines(text: str) -> list[str]:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if is_noise_line(line):
            continue

        lines.append(line)

    return lines


def create_chunks(documents: list[dict]) -> list[dict]:
    chunks = []

    chunk_id = 0
    current_section_by_source = {}

    for document in documents:
        source = document["source"]
        page = document["page"]
        text = document["text"]

        # 빈 페이지 제외
        if len(text) < MIN_PAGE_CHARS:
            continue

        # 목차 페이지 제외
        if is_toc_page(text):
            continue

        lines = clean_page_lines(text)

        if not lines:
            continue

        current_section = current_section_by_source.get(
            source,
            "General"
        )

        current_lines = []
        chunk_section = current_section

        def flush_chunk():
            nonlocal chunk_id
            nonlocal current_lines

            chunk_text = "\n".join(current_lines).strip()

            if len(chunk_text) < MIN_CHUNK_CHARS:
                return

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "source": source,
                    "page": page,
                    "section": chunk_section,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )

            chunk_id += 1

        for line in lines:

            # 새로운 section heading 발견
            if is_heading(line):
                if current_lines:
                    flush_chunk()

                current_section = line
                chunk_section = current_section
                current_lines = [line]

                continue

            candidate = "\n".join(
                current_lines + [line]
            )

            # 목표 길이 초과 시 기존 chunk 저장
            if (
                len(candidate) > TARGET_CHARS
                and current_lines
            ):
                flush_chunk()

                # arbitrary character가 아니라
                # 마지막 2개 line만 overlap
                current_lines = current_lines[
                    -OVERLAP_LINES:
                ]

                chunk_section = current_section

            current_lines.append(line)

        # 페이지 마지막 chunk
        if current_lines:
            flush_chunk()

        current_section_by_source[source] = current_section

    return chunks


if __name__ == "__main__":
    documents = load_all_documents()
    chunks = create_chunks(documents)

    print(f"Loaded pages  : {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    print("\n=== SAMPLE CHUNKS ===")

    for chunk in chunks[:15]:
        print("\n" + "=" * 80)

        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['source']} | "
            f"page={chunk['page']} | "
            f"section={chunk['section']} | "
            f"chars={chunk['char_count']}"
        )

        print("-" * 80)
        print(chunk["text"][:1200])