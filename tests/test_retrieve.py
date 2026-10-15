import pytest
from qdrant_client import QdrantClient, models

from app import retrieve as retrieve_module
from app.config import settings
from app.ingestion import vector_store
from app.retrieve import retrieve

COLLECTION = "test_collection"

# doc id -> (dense vector, sparse indices/values)
POINTS = {
    "a.md": ([1.0, 0.0, 0.0, 0.0], ([0], [1.0])),
    "b.md": ([0.0, 1.0, 0.0, 0.0], ([1], [1.0])),
    "c.md": ([0.7, 0.7, 0.0, 0.0], ([0, 1], [0.5, 0.5])),
    "d.md": ([0.0, 0.0, 1.0, 0.0], ([2], [1.0])),
}


@pytest.fixture
def client(monkeypatch):
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        COLLECTION,
        vectors_config={settings.DENSE_VECTOR_NAME: models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={settings.SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )
    qdrant.upsert(
        COLLECTION,
        [
            models.PointStruct(
                id=i,
                vector={
                    settings.DENSE_VECTOR_NAME: dense,
                    settings.SPARSE_VECTOR_NAME: models.SparseVector(indices=idx, values=vals),
                },
                payload={"page_content": f"text of {doc_id}", "metadata": {"doc_id": doc_id}},
            )
            for i, (doc_id, (dense, (idx, vals))) in enumerate(POINTS.items())
        ],
    )
    monkeypatch.setattr(settings, "qdrant_collection", COLLECTION)
    monkeypatch.setattr(retrieve_module, "get_client", lambda: qdrant)
    # The query is "about a.md": dense points along axis 0, sparse hits term 0.
    monkeypatch.setattr(retrieve_module, "embed_query_dense", lambda q: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        retrieve_module,
        "embed_query_sparse",
        lambda q: models.SparseVector(indices=[0], values=[1.0]),
    )
    return qdrant


def doc_ids(docs):
    return [d.metadata["doc_id"] for d in docs]


def test_dense_ranks_nearest_vector_first(client):
    docs = retrieve("q", k=3, mode="dense")
    assert doc_ids(docs)[0] == "a.md"
    assert doc_ids(docs)[1] == "c.md"


def test_sparse_ranks_keyword_overlap_first(client):
    docs = retrieve("q", k=3, mode="sparse")
    assert doc_ids(docs)[0] == "a.md"
    assert "b.md" not in doc_ids(docs)[:2]


def test_hybrid_fuses_and_scores(client):
    docs = retrieve("q", k=3, mode="hybrid")
    assert doc_ids(docs)[0] == "a.md"
    assert len(docs) == 3
    assert all("score" in d.metadata for d in docs)
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_documents_keep_content_and_metadata(client):
    (doc,) = retrieve("q", k=1, mode="dense")
    assert doc.page_content == "text of a.md"
    assert doc.metadata["doc_id"] == "a.md"


def test_k_override_and_default(client, monkeypatch):
    assert len(retrieve("q", k=2)) == 2
    monkeypatch.setattr(settings, "retrieval_k", 1)
    assert len(retrieve("q")) == 1


def test_invalid_mode_raises(client):
    with pytest.raises(ValueError, match="mode must be"):
        retrieve("q", mode="semantic")


def test_get_client_without_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "qdrant")
    vector_store.get_client.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="No index found"):
            vector_store.get_client()
    finally:
        vector_store.get_client.cache_clear()
