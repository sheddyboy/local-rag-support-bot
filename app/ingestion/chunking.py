from __future__ import annotations

import uuid

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

PRODUCT_DOC_SIZE = 2400  # about 600 tokens per section chunk
WHOLE_DOC_SIZE = 4000  # about 1000 tokens: tickets (max seen 2,924 chars) and runbook pages stay whole
MIN_CHUNK_CHARS = 50  # smaller chunks are usually a lone heading, so drop them

# Fixed namespace so a chunk gets the same ID on every run (re-ingesting upserts, never duplicates)
NAMESPACE = uuid.UUID("6f1b7a2e-3c1d-4a56-9f0e-5b7c2d9e1a10")

header_split = MarkdownHeaderTextSplitter(
    [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
)
size_split = RecursiveCharacterTextSplitter(
    chunk_size=PRODUCT_DOC_SIZE, chunk_overlap=PRODUCT_DOC_SIZE // 10
)
whole_split = RecursiveCharacterTextSplitter(
    chunk_size=WHOLE_DOC_SIZE, chunk_overlap=WHOLE_DOC_SIZE // 10
)


def chunk_documents(docs: list[Document]) -> list[Document]:
    if not docs:
        return []
    meta = docs[0].metadata

    if meta["source_type"] == "product_doc":
        sections = header_split.split_text(docs[0].page_content)
        for section in sections:
            section.metadata = {**meta, **section.metadata}  # keep doc_id, title, ...
        chunks = size_split.split_documents(
            sections
        )  # only splits sections that are too long
    else:
        chunks = whole_split.split_documents(
            docs
        )  # runbooks, tickets: whole unless very long

    # Drop lone-heading chunks, but never drop everything from a very short document
    chunks = [
        c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS
    ] or chunks
    return _add_labels_and_ids(chunks)


def _add_labels_and_ids(chunks: list[Document]) -> list[Document]:
    for index, chunk in enumerate(chunks):
        meta = chunk.metadata

        # 1. The header splitter leaves h1/h2/h3 keys. Merge them into one "section".
        headings = [meta.pop(key) for key in ("h1", "h2", "h3") if key in meta]
        if headings:
            meta["section"] = " > ".join(headings)

        # 2. Write a label at the top of the chunk text (this is what the LLM and the
        #    search indexes see, and where the doc_id to cite comes from).
        label = f"[{meta['doc_id']}] {meta['title']}"
        if "section" in meta:
            label += f"\nSection: {meta['section']}"
        chunk.page_content = f"{label}\n{chunk.page_content}"

        # 3. Number the chunk and give it an id that is the same on every run.
        meta["chunk_index"] = index
        meta["n_chunks"] = len(chunks)
        meta["chunk_id"] = str(uuid.uuid5(NAMESPACE, f"{meta['doc_id']}#{index}"))
    return chunks
