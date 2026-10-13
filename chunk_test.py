from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

SOURCE_TYPES = {
    "product-docs": "product_doc",
    "runbooks": "runbook",
    "tickets": "ticket",
}


# def show(docs):
#     for doc in docs:
#         page = doc.metadata.get("page", 1)
#         title = doc.metadata.get("title", "Untitled")

#         display(Markdown(f"### Page {page}: {title}"))
#         display(Markdown(f"**Metadata:** `{doc.metadata}`"))
#         display(Markdown(doc.page_content))
#         display(Markdown("---"))


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
            pages = [(None, _read_ticket(path))]
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
    # Note: MuPDF prints a harmless "Using RapidOCR and Tesseract..." line to the
    # console on every call. It is written at C level, so Python cannot silence it.
    docs = PyMuPDF4LLMLoader(str(path), mode="page").load()
    # The loader numbers pages from 0; people say "page 1", so add 1.
    return [
        (doc.metadata["page"] + 1, doc.page_content.replace("**", "").strip())
        for doc in docs
    ]


def _read_ticket(path: Path) -> str:
    """Turn the fixed ticket template into clean Markdown."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    for code in soup.select(".message-body code"):
        code.replace_with(f"`{code.get_text()}`")
    for item in soup.select(".message-body li"):
        item.insert_before("- ")

    ticket_no = (
        soup.select_one(".ticket-id").get_text(strip=True).replace("TICKET", "").strip()
    )
    lines = [f"# {ticket_no}: {soup.h1.get_text(strip=True)}", ""]

    # Header fields, skipping customer and agent names (noise for retrieval)
    for field in soup.select(".meta-row > span"):
        label = field.strong.extract().get_text(strip=True).rstrip(":")
        if label not in ("Customer", "Agent"):
            lines.append(f"- {label}: {field.get_text(strip=True)}")

    lines += ["", "## Conversation", ""]
    for message in soup.select(".message"):
        role = message.select_one(".role-badge").get_text(strip=True)
        when = message.select_one(".timestamp").get_text(strip=True)
        body = " ".join(
            message.select_one(".message-body").get_text(" ", strip=True).split()
        )
        body = re.sub(r"\s+([,.;:])", r"\1", body)  # "SSO ," -> "SSO,"
        lines += [f"**{role}** ({when}): {body}", ""]

    resolution = soup.select_one(".resolution p").get_text(" ", strip=True)
    lines += ["## Resolution", "", resolution]
    return "\n".join(lines)


def _title(text: str, path: Path) -> str:
    """First non-empty line, minus any leading '#'."""
    for line in text.splitlines():
        if line.strip():
            return line.lstrip("# ").strip()
    return path.stem.replace("-", " ")


root = Path("corpus")
header_split = MarkdownHeaderTextSplitter(
    [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
)
plain_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
md_split = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN, chunk_size=3000, chunk_overlap=300
)


def show(name, chunks):
    print(f"\n--- {name}: {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks):
        print(i, len(c.page_content), repr(c.page_content[:70]))


def try_splitters(path):
    docs = load_document(Path(path), root)
    text = "\n\n".join(d.page_content for d in docs)
    print(f"\n{path}: {len(docs)} doc(s), {len(text)} chars")
    show("MarkdownHeaderTextSplitter", header_split.split_text(text))
    show("RecursiveCharacterTextSplitter", plain_split.split_documents(docs))
    show("Recursive, Markdown-aware", md_split.split_documents(docs))


try_splitters("corpus/product-docs/billing/plans.md")
try_splitters("corpus/runbooks/engineering/deploy-rollback.pdf")
try_splitters("corpus/tickets/ticket-1042-sso-redirect-loop.html")
