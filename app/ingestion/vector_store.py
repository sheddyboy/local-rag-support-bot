"""The Qdrant vector store: rebuild it (ingestion) and open it (retrieval).

Embedded Qdrant keeps everything in a folder and lets only one process use it at a
time, so rebuild_collection() closes its connection when done, and whoever calls
open_vector_store() should call store.client.close() when finished.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from app.config import settings
from app.ingestion import embedding


def rebuild_collection(chunks: list[Document]) -> int:
    """Embed every chunk and rebuild the collection from scratch. Returns the point count."""
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


def open_vector_store(mode: RetrievalMode = RetrievalMode.HYBRID) -> QdrantVectorStore:
    """Open the existing collection for searching (used by retrieval).

    mode: HYBRID (default), DENSE or SPARSE. Any mode works on the same collection.
    """
    return QdrantVectorStore.from_existing_collection(
        collection_name=settings.qdrant_collection,
        embedding=embedding.get_dense_embeddings(),
        sparse_embedding=embedding.get_sparse_embeddings(),
        retrieval_mode=mode,
        vector_name=settings.DENSE_VECTOR_NAME,
        sparse_vector_name=settings.SPARSE_VECTOR_NAME,
        path=str(settings.qdrant_path),
    )
