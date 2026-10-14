"""Retrieval: a question in, the most relevant chunks out."""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from qdrant_client import models

from app.config import settings
from app.ingestion.embedding import embed_query_dense, embed_query_sparse
from app.ingestion.vector_store import get_client

Mode = Literal["hybrid", "dense", "sparse"]

# In hybrid mode each ranker (dense, BM25) first proposes this many candidates, and
# the fusion step picks the final top k from them.
CANDIDATES_PER_RANKER = 20


def retrieve(query: str, k: int | None = None, mode: Mode = "hybrid") -> list[Document]:
    """The top-k chunks for a question, best first.

    hybrid  dense + BM25, fused with reciprocal rank fusion (RRF). The default.
    dense   meaning only: the baseline we compare hybrid against.
    sparse  BM25 keywords only.

    Each chunk gets a "score" in its metadata. For hybrid this is an RRF score, so it
    only reflects the ranking, not similarity.
    """
    if mode == "hybrid":
        search = {
            "prefetch": [
                models.Prefetch(
                    query=embed_query_dense(query),
                    using=settings.DENSE_VECTOR_NAME,
                    limit=CANDIDATES_PER_RANKER,
                ),
                models.Prefetch(
                    query=embed_query_sparse(query),
                    using=settings.SPARSE_VECTOR_NAME,
                    limit=CANDIDATES_PER_RANKER,
                ),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
        }
    elif mode == "dense":
        search = {
            "query": embed_query_dense(query),
            "using": settings.DENSE_VECTOR_NAME,
        }
    elif mode == "sparse":
        search = {
            "query": embed_query_sparse(query),
            "using": settings.SPARSE_VECTOR_NAME,
        }
    else:
        raise ValueError(f"mode must be 'hybrid', 'dense' or 'sparse', not {mode!r}")

    points = (
        get_client()
        .query_points(
            settings.qdrant_collection,
            limit=k or settings.retrieval_k,
            with_payload=True,
            **search,
        )
        .points
    )

    docs: list[Document] = []
    for point in points:
        payload = point.payload or {}
        docs.append(
            Document(
                page_content=payload.get("page_content", ""),
                metadata={**(payload.get("metadata") or {}), "score": point.score},
            )
        )
    return docs
