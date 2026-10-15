import json
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from app.config import settings
from evals import harness, judges, metrics

REPO = Path(__file__).resolve().parent.parent


def fake_judge_llm(result):
    def run(_):
        if isinstance(result, Exception):
            raise result
        return result

    return RunnableLambda(run)


# ---------------------------------------------------------------- metrics
def test_context_precision_is_rank_aware():
    expected = {"a"}
    assert metrics.context_precision(["a", "x", "y"], expected) == 1.0
    assert metrics.context_precision(["x", "a", "y"], expected) == 0.5
    assert metrics.context_precision(["x", "y"], expected) == 0.0
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert metrics.context_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(5 / 6)


def test_context_recall_and_hit():
    assert metrics.context_recall(["a", "x"], {"a", "b"}) == 0.5
    assert metrics.context_recall(["a", "b"], {"a", "b"}) == 1.0
    assert metrics.context_hit(["x", "a"], {"a", "b"}) == 1.0
    assert metrics.context_hit(["x"], {"a"}) == 0.0


def test_citation_scores():
    assert metrics.citation_scores(["a", "x"], {"a", "b"}) == (0.5, 0.5)
    assert metrics.citation_scores(["a", "a"], {"a"}) == (1.0, 1.0)
    assert metrics.citation_scores([], {"a"}) == (None, 0.0)


# ---------------------------------------------------------------- judges
def test_faithfulness_ratio():
    verdict = judges.FaithfulnessVerdict(
        statements=[
            judges.Statement(statement="one", supported=True),
            judges.Statement(statement="two", supported=True),
            judges.Statement(statement="three", supported=False),
            judges.Statement(statement="four", supported=True),
        ]
    )
    result = judges.faithfulness("ans", ["ctx"], llm=fake_judge_llm(verdict))
    assert result["score"] == 0.75
    assert "three" in result["reason"]


def test_faithfulness_none_when_unparseable_or_empty():
    bad = judges.faithfulness("a", ["c"], llm=fake_judge_llm(OutputParserException("x")))
    assert bad["score"] is None
    empty = judges.faithfulness("a", ["c"], llm=fake_judge_llm(judges.FaithfulnessVerdict(statements=[])))
    assert empty["score"] is None


def test_answer_relevance_scores():
    llm = fake_judge_llm(judges.RelevanceCheck(questions=["q1", "q2"], noncommittal=False))
    same = lambda texts: [[1.0, 0.0] for _ in texts]
    assert judges.answer_relevance("q", "a", llm=llm, embed=same)["score"] == pytest.approx(1.0)

    opposite = lambda texts: [[1.0, 0.0]] + [[-1.0, 0.0] for _ in texts[1:]]
    assert judges.answer_relevance("q", "a", llm=llm, embed=opposite)["score"] == 0.0  # floored

    evasive = fake_judge_llm(judges.RelevanceCheck(questions=["q1"], noncommittal=True))
    assert judges.answer_relevance("q", "a", llm=evasive, embed=same)["score"] == 0.0


@pytest.mark.parametrize("verdict,score", [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0)])
def test_answer_correctness_mapping(verdict, score):
    llm = fake_judge_llm(judges.CorrectnessVerdict(verdict=verdict, reason="r"))
    result = judges.answer_correctness("q", "a", "truth", "notes", llm=llm)
    assert result == {"score": score, "reason": "r"}


def test_cosine_handles_zero_vector():
    assert judges._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert judges._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------- harness helpers
def test_mean_of_ignores_none():
    assert harness.mean_of([1.0, None, 0.0]) == (0.5, 2)
    assert harness.mean_of([None]) == (None, 0)


def test_latest_by_id_keeps_last_record():
    records = [{"id": "Q1", "v": 1}, {"id": "Q2", "v": 1}, {"id": "Q1", "v": 2}]
    assert harness.latest_by_id(records)["Q1"]["v"] == 2


def test_summarise_breakdown_and_failure_score():
    rows = [
        {"category": "x", "context_precision": 1.0, "faithfulness": None},
        {"category": "x", "context_precision": 0.5, "faithfulness": 1.0},
        {"category": "y", "context_precision": 0.0},
    ]
    summary = harness.summarise(rows)
    assert summary["context_precision"] == {"mean": 0.5, "n": 3}
    assert summary["faithfulness"] == {"mean": 1.0, "n": 1}
    assert summary["answer_correctness"] == {"mean": None, "n": 0}
    groups = harness.breakdown(rows, "category")
    assert groups["x"]["n_queries"] == 2 and groups["y"]["n_queries"] == 1
    assert harness.failure_score({"context_precision": 0.5, "faithfulness": 1.0}) == 0.75
    assert harness.failure_score({}) == 1.0


def test_load_test_set_limit_and_missing(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"queries": [{"id": "Q1"}, {"id": "Q2"}, {"id": "Q3"}]}))
    assert len(harness.load_test_set(path)) == 3
    assert [q["id"] for q in harness.load_test_set(path, limit=2)] == ["Q1", "Q2"]
    with pytest.raises(SystemExit):
        harness.load_test_set(tmp_path / "missing.json")


# ---------------------------------------------------------------- the real test set
def test_real_test_set_is_consistent_with_corpus():
    data = json.loads((REPO / "evals/test_set.json").read_text(encoding="utf-8"))
    queries = data["queries"]
    assert len(queries) == 50
    assert len({q["id"] for q in queries}) == 50
    required = {"id", "query", "ground_truth_answer", "expected_sources", "difficulty", "category"}
    corpus = REPO / "corpus"
    for q in queries:
        assert required <= q.keys(), q["id"]
        assert q["expected_sources"], q["id"]
        if corpus.exists():
            for source in q["expected_sources"]:
                assert (corpus / source).is_file(), f"{q['id']}: {source} not in corpus"


# ---------------------------------------------------------------- end to end with fakes
QUERIES = [
    {
        "id": "Q1",
        "query": "How many steps?",
        "ground_truth_answer": "100",
        "expected_sources": ["docs/a.md"],
        "difficulty": "easy",
        "category": "workflows",
        "notes": "",
    },
    {
        "id": "Q2",
        "query": "What does Pro cost?",
        "ground_truth_answer": "$299",
        "expected_sources": ["docs/b.md"],
        "difficulty": "hard",
        "category": "billing",
        "notes": "",
    },
]


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    test_set = tmp_path / "test_set.json"
    test_set.write_text(json.dumps({"queries": QUERIES}))
    monkeypatch.setattr(settings, "test_set_path", test_set)
    monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
    return tmp_path / "results"


class CountingPipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, question, k=None, mode="hybrid"):
        self.calls += 1
        doc = "docs/a.md" if "steps" in question else "docs/b.md"
        return {
            "question": question,
            "answer": "an answer",
            "sources": [doc],
            "confidence": "high",
            "retrieved_doc_ids": [doc],
            "chunk_doc_ids": [doc],
            "contexts": ["ctx"],
            "seconds": {"retrieve": 0.1, "generate": 0.2},
        }


def perfect_judge(item, record):
    return {
        "faithfulness": {"score": 1.0, "reason": "ok"},
        "answer_relevance": {"score": 0.9, "reason": "ok"},
        "answer_correctness": {"score": 1.0, "reason": "ok"},
    }


def test_harness_end_to_end_writes_report_and_passes_gates(run_env, capsys):
    pipeline = CountingPipeline()
    harness.main(["--run-id", "r1"], pipeline_fn=pipeline, judge_fn=perfect_judge)

    assert pipeline.calls == 2
    for name in ("r1/answers.jsonl", "r1/scores.jsonl", "r1/run_config.json", "r1.json", "r1.md"):
        assert (run_env / name).exists(), name
    report = json.loads((run_env / "r1.json").read_text())
    assert report["n_queries"] == 2
    assert report["metrics"]["context_precision"]["mean"] == 1.0
    assert report["thresholds"]["faithfulness"]["passed"] is True
    assert report["thresholds"]["context_precision"]["passed"] is True
    assert set(report["by_category"]) == {"billing", "workflows"}
    assert "Results:" in capsys.readouterr().out


def test_harness_resumes_without_recomputing(run_env):
    pipeline = CountingPipeline()
    harness.main(["--run-id", "r1"], pipeline_fn=pipeline, judge_fn=perfect_judge)
    harness.main(["--run-id", "r1"], pipeline_fn=pipeline, judge_fn=perfect_judge)
    assert pipeline.calls == 2
    scores = (run_env / "r1/scores.jsonl").read_text().splitlines()
    assert len(scores) == 2


def test_harness_no_judge_leaves_llm_metrics_empty(run_env):
    def explode(item, record):
        raise AssertionError("judge must not run with --no-judge")

    harness.main(["--run-id", "r2", "--no-judge"], pipeline_fn=CountingPipeline(), judge_fn=explode)
    report = json.loads((run_env / "r2.json").read_text())
    assert report["metrics"]["faithfulness"]["mean"] is None
    assert report["thresholds"]["faithfulness"]["passed"] is None
    assert report["metrics"]["context_hit"]["mean"] == 1.0


def test_harness_flags_failing_threshold(run_env):
    def weak_judge(item, record):
        return {**perfect_judge(item, record), "faithfulness": {"score": 0.2, "reason": "bad"}}

    harness.main(["--run-id", "r3"], pipeline_fn=CountingPipeline(), judge_fn=weak_judge)
    report = json.loads((run_env / "r3.json").read_text())
    assert report["thresholds"]["faithfulness"]["passed"] is False
