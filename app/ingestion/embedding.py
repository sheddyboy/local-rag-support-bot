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
from qdrant_client import models

from app.config import settings


@lru_cache
def get_dense_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.dense_model, base_url=settings.ollama_base_url
    )


@lru_cache
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=settings.sparse_model)


def embed_query_dense(query: str) -> list[float]:
    """Dense vector for a search query.

    Qwen3-Embedding is instruction aware: a query gets a one-line task instruction,
    documents never do (ingestion embeds documents as they are).
    """
    if (
        settings.use_query_instruction
    ):  # read on every call, so it can be toggled for A/B tests
        query = f"Instruct: {settings.query_instruction}\nQuery:{query}"
    return get_dense_embeddings().embed_query(query)


def embed_query_sparse(query: str) -> models.SparseVector:
    """BM25 vector for a search query. It gets the raw query, never the instruction."""
    vector = get_sparse_embeddings().embed_query(query)
    return models.SparseVector(indices=vector.indices, values=vector.values)
