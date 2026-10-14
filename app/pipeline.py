"""End to end: a question in, an answer with its sources out.

python -m app.pipeline "How do I roll back a bad deploy?"
python -m app.pipeline "How do I roll back a bad deploy?" --json
"""

from __future__ import annotations

import argparse
import json
import time

from langchain_core.runnables import Runnable

from app.generate import generate
from app.ingestion.vector_store import close_client
from app.retrieve import Mode, retrieve


def run_pipeline(
    question: str,
    k: int | None = None,
    mode: Mode = "hybrid",
    llm: Runnable | None = None,
) -> dict:
    """Retrieve, then generate. Returns everything the eval harness needs, as plain data."""
    start = time.perf_counter()
    chunks = retrieve(question, k=k, mode=mode)
    retrieved = time.perf_counter()
    result = generate(question, chunks, llm=llm)
    done = time.perf_counter()

    return {
        "question": question,
        "answer": result.answer,
        "sources": result.sources,  # the documents the model says it used (checked)
        "confidence": result.confidence,
        "retrieved_doc_ids": list(dict.fromkeys(c.metadata["doc_id"] for c in chunks)),
        "chunk_doc_ids": [
            c.metadata["doc_id"] for c in chunks
        ],  # one per chunk, in rank order
        "contexts": [c.page_content for c in chunks],  # what the model was shown
        "seconds": {
            "retrieve": round(retrieved - start, 2),
            "generate": round(done - retrieved, 2),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ask the support assistant a question."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--k", type=int, help="chunks to retrieve (default from settings)"
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default="hybrid"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(args.question, k=args.k, mode=args.mode)
    except RuntimeError as exc:  # for example: no index yet
        raise SystemExit(str(exc))
    finally:
        close_client()  # release the Qdrant folder lock

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{result['answer']}\n")
    print("Sources:")
    for source in result["sources"] or ["(none)"]:
        print(f"  - {source}")
    print(f"Confidence: {result['confidence']}")
    seconds = result["seconds"]
    print(
        f"\n(retrieved {len(result['contexts'])} chunks from {len(result['retrieved_doc_ids'])} "
        f"documents: {seconds['retrieve']}s retrieve, {seconds['generate']}s generate)"
    )


if __name__ == "__main__":
    main()
