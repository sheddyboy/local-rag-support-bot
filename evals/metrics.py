"""Deterministic retrieval and citation metrics. No LLM is involved.

A retrieved chunk counts as relevant if its document is one of the query's
expected_sources, so these are document-level labels.
"""

from __future__ import annotations


def context_precision(chunk_doc_ids: list[str], expected: set[str]) -> float:
    """Rank-aware precision, as RAGAS defines it.

    Take precision@k at every rank k where a relevant chunk appears, and average those
    values. A single relevant chunk at rank 1 scores 1.0, at rank 2 scores 0.5, at rank 3
    scores 0.33. It depends on WHERE the relevant chunks rank, not on how many you retrieve.
    """
    relevant_so_far = 0
    total = 0.0
    for rank, doc_id in enumerate(chunk_doc_ids, start=1):
        if doc_id in expected:
            relevant_so_far += 1
            total += relevant_so_far / rank
    return total / relevant_so_far if relevant_so_far else 0.0


def context_recall(chunk_doc_ids: list[str], expected: set[str]) -> float:
    """Share of the expected documents that were retrieved."""
    return len(expected & set(chunk_doc_ids)) / len(expected)


def context_hit(chunk_doc_ids: list[str], expected: set[str]) -> float:
    """1.0 if at least one expected document was retrieved, else 0.0."""
    return 1.0 if expected & set(chunk_doc_ids) else 0.0


def citation_scores(cited: list[str], expected: set[str]) -> tuple[float | None, float]:
    """(precision, recall) of the sources the model cited against the expected sources.

    Precision is None when nothing was cited, because it is undefined (0 / 0).
    """
    correct = len(set(cited) & expected)
    precision = correct / len(set(cited)) if cited else None
    return precision, correct / len(expected)
