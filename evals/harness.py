"""Evaluation harness.

    python -m evals.harness                       # answers, scores and report for every query
    python -m evals.harness --limit 3             # quick check on the first 3 queries
    python -m evals.harness --no-judge            # retrieval and citation metrics only (fast)
    python -m evals.harness --run-id v1 --stage scores   # score a saved run again

Stages (each one resumes where it stopped, so a crash never loses finished work):
  answers  run the pipeline on every query      -> evals/results/<run_id>/answers.jsonl
  scores   compute the metrics for every answer -> evals/results/<run_id>/scores.jsonl
  report   aggregate                            -> evals/results/<run_id>.json and <run_id>.md

Run the stages in one go, or separately: the answers stage uses the generator model and the
scores stage uses the judge model, so running them apart keeps the two models from being
loaded in memory together.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingestion import chunking
from app.pipeline import run_pipeline
from evals import judges, metrics

THRESHOLDS = {"faithfulness": 0.70, "context_precision": 0.60}  # the brief's pass marks

LABELS = {
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "context_hit": "Context hit rate",
    "citation_precision": "Citation precision",
    "citation_recall": "Citation recall",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "answer_correctness": "Answer correctness",
}
METRICS = list(LABELS)

DEFINITIONS = """\
- **Context precision**: rank-aware precision (the RAGAS definition) of the retrieved chunks. A chunk is relevant if its document is in the query's expected sources. Not LLM-judged.
- **Context recall**: share of the expected source documents that were retrieved. Not LLM-judged.
- **Context hit rate**: share of queries where at least one expected document was retrieved. Not LLM-judged.
- **Citation precision / recall**: the documents the model cited, compared with the expected sources. Not LLM-judged.
- **Faithfulness**: share of the answer's statements that the retrieved context supports (RAGAS definition, LLM-judged).
- **Answer relevance**: how well the answer addresses the question (RAGAS definition, LLM-judged; an evasive answer scores 0).
- **Answer correctness**: the answer graded against the ground-truth answer and key facts in the query's notes, as correct (1), partial (0.5) or incorrect (0). LLM-judged.
"""


# ---------------------------------------------------------------- files
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def latest_by_id(records: list[dict]) -> dict[str, dict]:
    return {
        record["id"]: record for record in records
    }  # a later line replaces an earlier one


def update_run_config(run_dir: Path, **updates) -> dict:
    path = run_dir / "run_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    config.update(updates)
    path.write_text(json.dumps(config, indent=2))
    return config


def load_test_set(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Test set not found at {path}. Copy test_set.json there, or set TEST_SET_PATH."
        )
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    return queries[:limit] if limit else queries


# ---------------------------------------------------------------- stage 1: answers
def answer_queries(
    queries: list[dict], run_dir: Path, pipeline_fn: Callable, k: int | None, mode: str
) -> None:
    path = run_dir / "answers.jsonl"
    done = {record["id"] for record in read_jsonl(path)}
    for number, item in enumerate(queries, start=1):
        if item["id"] in done:
            continue
        print(f"[answers {number}/{len(queries)}] {item['id']}  {item['query'][:70]}")
        append_jsonl(
            path, {"id": item["id"], **pipeline_fn(item["query"], k=k, mode=mode)}
        )


# ---------------------------------------------------------------- stage 2: scores
def retrieval_scores(item: dict, record: dict) -> dict:
    """The deterministic metrics: no LLM involved."""
    expected = set(item["expected_sources"])
    citation_precision, citation_recall = metrics.citation_scores(
        record["sources"], expected
    )
    return {
        "context_precision": metrics.context_precision(
            record["chunk_doc_ids"], expected
        ),
        "context_recall": metrics.context_recall(record["chunk_doc_ids"], expected),
        "context_hit": metrics.context_hit(record["chunk_doc_ids"], expected),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


def score_answers(
    queries: list[dict], run_dir: Path, judge_fn: Callable | None
) -> None:
    answers = latest_by_id(read_jsonl(run_dir / "answers.jsonl"))
    path = run_dir / "scores.jsonl"
    # A query counts as done if it is fully scored for this mode (with or without the judge)
    done = {r["id"] for r in read_jsonl(path) if r.get("judged") or judge_fn is None}
    for number, item in enumerate(queries, start=1):
        if item["id"] in done or item["id"] not in answers:
            continue
        print(f"[scores {number}/{len(queries)}] {item['id']}")
        record = answers[item["id"]]
        scores, notes = retrieval_scores(item, record), {}
        if judge_fn is not None:
            for name, result in judge_fn(item, record).items():
                scores[name], notes[name] = result["score"], result["reason"]
        append_jsonl(
            path,
            {
                "id": item["id"],
                "judged": judge_fn is not None,
                **scores,
                "notes": notes,
            },
        )


# ---------------------------------------------------------------- stage 3: report
def mean_of(values) -> tuple[float | None, int]:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present), len(present)) if present else (None, 0)


def summarise(rows: list[dict]) -> dict:
    summary = {}
    for name in METRICS:
        average, count = mean_of(row.get(name) for row in rows)
        summary[name] = {
            "mean": None if average is None else round(average, 3),
            "n": count,
        }
    return summary


def breakdown(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return {
        group: {"n_queries": len(members), "metrics": summarise(members)}
        for group, members in sorted(groups.items())
    }


def failure_score(row: dict) -> float:
    """Mean of the main scores a query has; used to rank the weakest queries."""
    average, _ = mean_of(
        row.get(name)
        for name in ("context_precision", "faithfulness", "answer_correctness")
    )
    return 1.0 if average is None else average


def build_report(run_id: str, queries: list[dict], run_dir: Path) -> dict[str, Any]:
    answers = latest_by_id(read_jsonl(run_dir / "answers.jsonl"))
    scores = latest_by_id(read_jsonl(run_dir / "scores.jsonl"))
    config = (
        json.loads((run_dir / "run_config.json").read_text())
        if (run_dir / "run_config.json").exists()
        else {}
    )

    rows = []
    for item in queries:
        if item["id"] not in scores or item["id"] not in answers:
            continue
        answer, score = answers[item["id"]], scores[item["id"]]
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "expected_sources": item["expected_sources"],
                "ground_truth_answer": item["ground_truth_answer"],
                "answer": answer["answer"],
                "sources": answer["sources"],
                "confidence": answer["confidence"],
                "retrieved_doc_ids": answer["retrieved_doc_ids"],
                "seconds": answer["seconds"],
                **{name: score.get(name) for name in METRICS},
                "judge_notes": score.get("notes", {}),
            }
        )
    if not rows:
        raise SystemExit("Nothing to report: run the answers and scores stages first.")

    overall = summarise(rows)
    thresholds = {}
    for name, required in THRESHOLDS.items():
        value = overall[name]["mean"]
        thresholds[name] = {
            "required": required,
            "score": value,
            "passed": None if value is None else value >= required,
        }

    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "n_queries": len(rows),
        "metrics": overall,
        "thresholds": thresholds,
        "latency_seconds": {
            "retrieve_mean": round(
                mean_of(r["seconds"]["retrieve"] for r in rows)[0] or 0.0, 2
            ),
            "generate_mean": round(
                mean_of(r["seconds"]["generate"] for r in rows)[0] or 0.0, 2
            ),
        },
        "by_difficulty": breakdown(rows, "difficulty"),
        "by_category": breakdown(rows, "category"),
        "queries": rows,
    }


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(report: dict, failures: int = 5) -> str:
    config = report["config"]
    lines = [
        f"# Evaluation run {report['run_id']}",
        "",
        f"{report['n_queries']} queries, generated {report['created_at']}.",
        f"Generator `{config.get('generation_model')}`, judge `{config.get('judge_model')}` ",
        f"embeddings `{config.get('embedding_model')}`. Retrieval `{config.get('retrieval_mode')}` ",
        f"k={config.get('k')}.",
        "",
        "## Results",
        "",
        "| Metric | Score | Scored (n) | Required | Result |",
        "|---|---|---|---|---|",
    ]
    for name in METRICS:
        entry = report["metrics"][name]
        gate = report["thresholds"].get(name)
        required = f">= {gate['required']:.2f}" if gate else ""
        result = (
            ""
            if not gate
            else (
                "not scored"
                if gate["passed"] is None
                else ("PASS" if gate["passed"] else "FAIL")
            )
        )
        lines.append(
            f"| {LABELS[name]} | {fmt(entry['mean'])} | {entry['n']} | {required} | {result} |"
        )

    latency = report["latency_seconds"]
    lines += [
        "",
        f"Mean time per query: {latency['retrieve_mean']}s retrieving, {latency['generate_mean']}s generating.",
        "",
    ]

    columns = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevance",
        "answer_correctness",
    ]
    for title, key in (
        ("By difficulty", "by_difficulty"),
        ("By category", "by_category"),
    ):
        lines += [
            f"## {title}",
            "",
            "| Group | Queries | " + " | ".join(LABELS[c] for c in columns) + " |",
            "|---|---|" + "---|" * len(columns),
        ]
        for group, data in report[key].items():
            cells = " | ".join(fmt(data["metrics"][c]["mean"]) for c in columns)
            lines.append(f"| {group} | {data['n_queries']} | {cells} |")
        lines.append("")

    lines += [f"## Weakest {failures} queries", ""]
    for row in sorted(report["queries"], key=failure_score)[:failures]:
        shown = {name: row[name] for name in METRICS if row[name] is not None}
        lines += [
            f"### {row['id']} ({row['category']}, {row['difficulty']})",
            "",
            f"- **Question:** {row['query']}",
            f"- **Expected sources:** {', '.join(row['expected_sources'])}",
            f"- **Retrieved:** {', '.join(row['retrieved_doc_ids'])}",
            f"- **Cited:** {', '.join(row['sources']) or '(none)'} (confidence {row['confidence']})",
            f"- **Answer:** {row['answer'][:400]}",
            f"- **Ground truth:** {row['ground_truth_answer']}",
            f"- **Scores:** {', '.join(f'{k} {v:.2f}' for k, v in shown.items())}",
            *[
                f"- **Judge ({name}):** {note[:300]}"
                for name, note in row["judge_notes"].items()
            ],
            "",
        ]
    lines += ["## Metric definitions", "", DEFINITIONS]
    return "\n".join(lines)


def write_report(run_id: str, queries: list[dict], run_dir: Path) -> dict[str, Any]:
    report = build_report(run_id, queries, run_dir)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / f"{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (settings.results_dir / f"{run_id}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------- command line
def main(
    argv: list[str] | None = None,
    pipeline_fn: Callable = run_pipeline,
    judge_fn: Callable | None = judges.judge_all,
) -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation set and report the metrics."
    )
    parser.add_argument("--limit", type=int, help="only the first N queries")
    parser.add_argument(
        "--run-id",
        help="name of the run (default: a timestamp). Reuse a name to resume it.",
    )
    parser.add_argument(
        "--stage", choices=["all", "answers", "scores", "report"], default="all"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="skip the LLM-judged metrics"
    )
    parser.add_argument(
        "--k", type=int, help="chunks to retrieve (default from settings)"
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default="hybrid"
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = settings.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    queries = load_test_set(settings.test_set_path, args.limit)
    judge = None if args.no_judge else judge_fn

    if args.stage in ("all", "answers"):
        update_run_config(
            run_dir,
            generation_model=settings.generation_model,
            embedding_model=settings.dense_model,
            retrieval_mode=args.mode,
            k=args.k or settings.retrieval_k,
            num_ctx=settings.num_ctx,
            chunk_size_product_doc=chunking.PRODUCT_DOC_SIZE,
            chunk_size_whole_doc=chunking.WHOLE_DOC_SIZE,
        )
        answer_queries(queries, run_dir, pipeline_fn, args.k, args.mode)
    if args.stage in ("all", "scores"):
        update_run_config(
            run_dir, judge_model=None if judge is None else settings.judge_model
        )
        score_answers(queries, run_dir, judge)
    if args.stage in ("all", "report"):
        report = write_report(run_id, queries, run_dir)
        print(
            f"\nResults: {settings.results_dir / (run_id + '.json')}\nSummary: {settings.results_dir / (run_id + '.md')}\n"
        )
        for name in METRICS:
            entry = report["metrics"][name]
            print(f"  {LABELS[name]:20} {fmt(entry['mean'])}  (n={entry['n']})")
        for name, gate in report["thresholds"].items():
            state = (
                "not scored"
                if gate["passed"] is None
                else ("PASS" if gate["passed"] else "FAIL")
            )
            print(f"  {name} >= {gate['required']:.2f}: {state}")


if __name__ == "__main__":
    main()
