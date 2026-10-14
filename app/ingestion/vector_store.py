"""The Qdrant vector store: rebuild it (ingestion) and open it (retrieval).

Every chunk is stored with two vectors, a dense one (meaning) and a sparse BM25 one
(keywords). Because both are stored, retrieval can search in hybrid, dense-only or
sparse-only mode, which lets us measure hybrid against a dense baseline.

Embedded Qdrant keeps everything in a folder and lets only one connection use it at a
time. Retrieval shares ONE client (get_client), and rebuild_collection() closes it
first. In a notebook, call close_client() before running anything else that opens
the same folder.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from app.config import settings
from app.ingestion import embedding


def rebuild_collection(chunks: list[Document]) -> int:
    """Embed every chunk and rebuild the collection from scratch. Returns the point count."""
    close_client()  # release the folder if retrieval had it open in this process
    store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embedding.get_dense_embeddings(),
        sparse_embedding=embedding.get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=settings.DENSE_VECTOR_NAME,
        sparse_vector_name=settings.SPARSE_VECTOR_NAME,
        path=str(settings.qdrant_path),
        collection_name=settings.qdrant_collection,
        ids=[c.metadata["chunk_id"] for c in chunks],
        batch_size=32,
        force_recreate=True,  # rebuild from scratch on every run
    )
    count = store.client.count(settings.qdrant_collection).count
    store.client.close()
    return count


@lru_cache
def get_client() -> QdrantClient:
    """The one shared connection to the embedded Qdrant folder (used by retrieval)."""
    client = QdrantClient(path=str(settings.qdrant_path))
    if not client.collection_exists(settings.qdrant_collection):
        client.close()  # release the folder before raising
        raise RuntimeError(
            f"No index found at {settings.qdrant_path}. Run: python -m ingestion.main"
        )
    return client


def close_client() -> None:
    """Release the Qdrant folder lock."""
    if get_client.cache_info().currsize:  # only if a client was actually opened
        get_client().close()
    get_client.cache_clear()
