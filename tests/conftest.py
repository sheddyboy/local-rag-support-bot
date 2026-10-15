"""Shared fixtures. Tests run fully offline: no Ollama, no model downloads, no built index."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


def make_chunk(doc_id: str, text: str = "some text", **meta) -> Document:
    metadata = {"doc_id": doc_id, "title": doc_id, "source_type": "product_doc", **meta}
    return Document(page_content=f"[{doc_id}] {text}", metadata=metadata)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any code path that would reach Ollama fails the test instead of hanging or passing silently."""

    def blocked(*args, **kwargs):
        raise AssertionError("test tried to reach Ollama; inject a fake instead")

    monkeypatch.setattr("app.ingestion.embedding.get_dense_embeddings", blocked)
    monkeypatch.setattr("app.generate.get_llm", blocked)
    monkeypatch.setattr("evals.judges.get_judge", blocked)
    monkeypatch.setattr("evals.judges.get_dense_embeddings", blocked)
