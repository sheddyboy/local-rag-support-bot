"""Find the corpus files to ingest."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.loader import SOURCE_TYPES

SUPPORTED = {".md", ".pdf", ".html"}


def crawl_corpus(corpus_root: Path) -> list[Path]:
    """All supported files under the three corpus folders, in a stable order.

    Walking only product-docs/, runbooks/ and tickets/ means the README, test_set.json
    and any stray top-level folders are never picked up. Hidden files (.DS_Store and
    macOS "._" copies) are skipped.
    """
    files = []
    for folder in SOURCE_TYPES:
        for path in (corpus_root / folder).rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED
                and not path.name.startswith(".")
            ):
                files.append(path)
    return sorted(files)
