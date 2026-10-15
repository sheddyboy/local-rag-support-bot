import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.crawl_corpus import crawl_corpus
from app.ingestion import chunking, loader
from app.ingestion import main as ingest_main
from app.ingestion.chunking import chunk_documents
from app.ingestion.loader import load_document

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "corpus"


def write(path: Path, text: str = "# Title\n\nbody text that is long enough to keep.\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def product_doc(content: str, doc_id="product-docs/a/b.md") -> list[Document]:
    return [
        Document(
            page_content=content,
            metadata={"doc_id": doc_id, "source_type": "product_doc", "title": "Doc"},
        )
    ]


# ---------------------------------------------------------------- crawl
def test_crawl_finds_only_supported_files_in_source_folders(tmp_path):
    keep = [
        write(tmp_path / "product-docs/x/a.md"),
        write(tmp_path / "runbooks/r/b.pdf", "pdf"),
        write(tmp_path / "tickets/t.html", "<html/>"),
    ]
    write(tmp_path / "product-docs/.DS_Store", "x")
    write(tmp_path / "product-docs/x/._a.md")
    write(tmp_path / "product-docs/x/notes.txt")
    write(tmp_path / "README.md")
    write(tmp_path / "other/c.md")

    assert crawl_corpus(tmp_path) == sorted(keep)


# ---------------------------------------------------------------- loader
def test_load_markdown_metadata(tmp_path):
    path = write(tmp_path / "product-docs/billing/plans.md", "# Plans\n\nPro costs money.\n")
    (doc,) = load_document(path, tmp_path)
    assert doc.metadata == {
        "doc_id": "product-docs/billing/plans.md",
        "source_type": "product_doc",
        "title": "Plans",
    }


def test_load_rejects_unknown_folder_and_suffix(tmp_path):
    with pytest.raises(ValueError, match="top-level folder"):
        load_document(write(tmp_path / "misc/a.md"), tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(write(tmp_path / "product-docs/a.txt"), tmp_path)


def test_load_pdf_numbers_pages_from_one(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_read_pdf", lambda path: [(1, "# Runbook\nstep"), (2, "more")])
    docs = load_document(write(tmp_path / "runbooks/eng/x.pdf", "x"), tmp_path)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert {d.metadata["title"] for d in docs} == {"Runbook"}
    assert docs[0].metadata["source_type"] == "runbook"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus folder not present")
def test_load_ticket_html_becomes_clean_markdown():
    path = CORPUS / "tickets/ticket-1042-sso-redirect-loop.html"
    (doc,) = load_document(path, CORPUS)
    text = doc.page_content
    assert doc.metadata["ticket_id"] == "HX-1042"
    assert doc.metadata["source_type"] == "ticket"
    assert "## Conversation" in text and "## Resolution" in text
    assert "- Customer:" not in text and "- Agent:" not in text
    assert "- Priority: P1" in text


# ---------------------------------------------------------------- chunking
def test_product_doc_splits_on_headings_and_labels_chunks():
    content = "# Guide\n\nIntro paragraph that is long enough.\n\n## Setup\n\nSetup steps go here, long enough.\n"
    chunks = chunk_documents(product_doc(content))
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.page_content.startswith("[product-docs/a/b.md] Doc")
    setup = next(c for c in chunks if "Setup steps" in c.page_content)
    assert setup.metadata["section"] == "Guide > Setup"
    assert "Section: Guide > Setup" in setup.page_content
    assert "h1" not in setup.metadata and "h2" not in setup.metadata


def test_oversize_section_is_split():
    body = "word " * 1500  # about 7500 chars under one heading
    chunks = chunk_documents(product_doc(f"# Big\n\n{body}"))
    assert len(chunks) > 1
    assert all(len(c.page_content) < chunking.PRODUCT_DOC_SIZE + 400 for c in chunks)


def test_runbook_and_ticket_stay_whole():
    for source_type in ("runbook", "ticket"):
        doc = Document(
            page_content="Heading\n\n" + "line of text. " * 100,
            metadata={"doc_id": f"{source_type}s/x", "source_type": source_type, "title": "T"},
        )
        assert len(chunk_documents([doc])) == 1


def test_short_document_is_not_emptied_and_empty_input_is_empty():
    chunks = chunk_documents(product_doc("# Hi"))
    assert len(chunks) == 1
    assert chunk_documents([]) == []


def test_chunk_ids_are_stable_and_unique():
    content = "# A\n\n" + "alpha " * 20 + "\n\n## B\n\n" + "beta " * 20
    first = chunk_documents(product_doc(content))
    second = chunk_documents(product_doc(content))
    ids = [c.metadata["chunk_id"] for c in first]
    assert ids == [c.metadata["chunk_id"] for c in second]
    assert len(set(ids)) == len(ids)
    assert [c.metadata["chunk_index"] for c in first] == list(range(len(first)))
    assert {c.metadata["n_chunks"] for c in first} == {len(first)}


# ---------------------------------------------------------------- ingestion main
def test_collect_chunks_records_failures_without_aborting(tmp_path):
    write(tmp_path / "product-docs/ok.md", "# Ok\n\n" + "fine text " * 20)
    write(tmp_path / "tickets/bad.html", "<html><body>no ticket id</body></html>")
    write(tmp_path / "product-docs/empty.md", "")

    files, chunks, failed, no_chunks = ingest_main.collect_chunks(tmp_path)

    assert len(files) == 3
    assert len(chunks) >= 1
    assert [f["file"] for f in failed] == ["tickets/bad.html"]
    assert "ValueError" in failed[0]["error"]
    assert no_chunks == ["product-docs/empty.md"]


def test_main_no_embed_writes_report_without_indexing(tmp_path, monkeypatch):
    write(tmp_path / "corpus/product-docs/ok.md", "# Ok\n\n" + "fine text " * 20)
    report_path = tmp_path / "out/report.json"
    monkeypatch.setattr(settings, "ingestion_report_path", report_path)

    def boom(chunks):
        raise AssertionError("rebuild_collection must not run with --no-embed")

    monkeypatch.setattr(ingest_main, "rebuild_collection", boom)

    ingest_main.main(["--corpus", str(tmp_path / "corpus"), "--no-embed"])

    report = json.loads(report_path.read_text())
    assert report["files_found"] == 1
    assert report["files_failed"] == []
    assert report["chunks_by_source_type"] == {"product_doc": report["chunks_total"]}
    assert report["points_in_qdrant"] is None
