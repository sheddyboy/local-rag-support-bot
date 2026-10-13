"""Ingest the corpus.   python -m ingestion.main --corpus ./corpus

crawl -> load -> chunk (every file) -> embed (dense + BM25) -> rebuild the Qdrant collection
Add --no-embed to stop after chunking (no Ollama needed) and just write the report.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.crawl_corpus import crawl_corpus
from app.ingestion.chunking import chunk_documents
from app.ingestion.loader import load_document
from app.ingestion.vector_store import rebuild_collection


def collect_chunks(corpus: Path):
    """Load and chunk every file. One bad file is recorded, never fatal."""
    files = crawl_corpus(corpus)
    chunks: list[Document] = []
    failed: list[dict] = []
    no_chunks: list[str] = []
    for path in files:
        name = path.relative_to(corpus).as_posix()
        try:
            file_chunks = chunk_documents(load_document(path, corpus))
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not file_chunks:
            no_chunks.append(name)  # e.g. a scanned PDF with no text layer
        chunks += file_chunks
    return files, chunks, failed, no_chunks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=settings.corpus_dir)
    parser.add_argument("--no-embed", action="store_true", help="chunk and report only")
    args = parser.parse_args(argv)

    start = time.time()
    files, chunks, failed, no_chunks = collect_chunks(args.corpus)

    points = None
    if not args.no_embed and chunks:
        try:
            points = rebuild_collection(chunks)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Embedding or indexing failed: {type(exc).__name__}: {exc}\n"
                "Is Ollama running, and have you run: ollama pull qwen3-embedding:0.6b ?"
            )

    report = {
        "files_found": len(files),
        "files_failed": failed,
        "files_with_no_chunks": no_chunks,
        "chunks_total": len(chunks),
        "chunks_by_source_type": dict(
            Counter(c.metadata["source_type"] for c in chunks)
        ),
        "points_in_qdrant": points,
        "seconds": round(time.time() - start, 1),
    }
    settings.ingestion_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ingestion_report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
