"""The two embedding models, shared by ingestion and retrieval.

  dense   Ollama (qwen3-embedding)  -> meaning
  sparse  fastembed BM25            -> exact keywords (hybrid search)

They are functions, not module-level objects, so importing this file never
connects to Ollama or downloads a model.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse

from app.config import settings


@lru_cache
def get_dense_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.dense_model, base_url=settings.ollama_base_url
    )


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=settings.sparse_model)
