from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

SOURCE_TYPES = {
    "product-docs": "product_doc",
    "runbooks": "runbook",
    "tickets": "ticket",
}


def show(docs):
    for doc in docs:
        page = doc.metadata.get("page", 1)
        title = doc.metadata.get("title", "Untitled")

        display(Markdown(f"### Page {page}: {title}"))
        display(Markdown(f"**Metadata:** `{doc.metadata}`"))
        display(Markdown(doc.page_content))
        display(Markdown("---"))


def load_document(path: Path, corpus_root: Path) -> list[Document]:
    rel = path.relative_to(corpus_root)
    if rel.parts[0] not in SOURCE_TYPES:
        raise ValueError(
            f"{rel}: top-level folder must be one of {sorted(SOURCE_TYPES)}"
        )

    # Each reader returns a list of (page_number, text). page_number is None
    # for formats that have no pages.
    match path.suffix.lower():
        case ".md":
            pages = [(None, path.read_text(encoding="utf-8"))]
        case ".pdf":
            pages = _read_pdf(path)
        case ".html":
            pages = [(None, _read_html(path))]
        case other:
            raise ValueError(f"Unsupported file type {other!r}: {path}")

    metadata = {
        "doc_id": rel.as_posix(),  # matches expected_sources in test_set.json
        "source_type": SOURCE_TYPES[rel.parts[0]],
        "title": _title(pages[0][1], path),  # same title on every page
    }
    if metadata["source_type"] == "ticket":
        number = re.search(r"ticket-(\d+)-", path.name)
        if number:
            metadata["ticket_id"] = f"HX-{number.group(1)}"

    return [
        Document(
            page_content=text,
            metadata={**metadata, **({"page": number} if number else {})},
        )
        for number, text in pages
    ]


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    docs = PyMuPDF4LLMLoader(
        str(path),
        mode="page",
        use_ocr=True,
    ).load()
    # The loader numbers pages from 0; people say "page 1", so add 1.
    return [
        (doc.metadata["page"] + 1, doc.page_content.replace("**", "").strip())
        for doc in docs
    ]


def _read_html(path: Path) -> str:
    """Turn the fixed ticket template into clean Markdown."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    for code in soup.select(".message-body code"):
        code.replace_with(f"`{code.get_text()}`")
    for item in soup.select(".message-body li"):
        item.insert_before("- ")

    ticket_id = soup.select_one(".ticket-id")
    if ticket_id is None:
        raise ValueError(f"Missing .ticket-id element in {path}")
    ticket_no = ticket_id.get_text(strip=True).replace("TICKET", "").strip()

    title = soup.h1.get_text(strip=True) if soup.h1 is not None else "Untitled ticket"
    lines = [f"# {ticket_no}: {title}", ""]

    # Header fields, skipping customer and agent names (noise for retrieval)
    for field in soup.select(".meta-row > span"):
        label_elem = field.select_one("strong")
        if label_elem is None:
            continue

        label = label_elem.extract().get_text(strip=True).rstrip(":")
        if label not in ("Customer", "Agent"):
            lines.append(f"- {label}: {field.get_text(strip=True)}")

    lines += ["", "## Conversation", ""]
    for message in soup.select(".message"):
        role_badge = message.select_one(".role-badge")
        timestamp = message.select_one(".timestamp")
        body_elem = message.select_one(".message-body")
        if role_badge is None or timestamp is None or body_elem is None:
            continue

        role = role_badge.get_text(strip=True)
        when = timestamp.get_text(strip=True)
        body = " ".join(body_elem.get_text(" ", strip=True).split())
        body = re.sub(r"\s+([,.;:])", r"\1", body)  # "SSO ," -> "SSO,"
        lines += [f"**{role}** ({when}): {body}", ""]

    resolution = soup.select_one(".resolution p")
    if resolution is None:
        raise ValueError(f"Missing .resolution p element in {path}")

    lines += ["## Resolution", "", resolution.get_text(" ", strip=True)]
    return "\n".join(lines)


def _title(text: str, path: Path) -> str:
    """First non-empty line, minus any leading '#'."""
    for line in text.splitlines():
        if line.strip():
            return line.lstrip("# ").strip()
    return path.stem.replace("-", " ")
