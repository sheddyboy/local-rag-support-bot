"""LLM-judged metrics: faithfulness, answer relevance, answer correctness.

Each function returns {"score": float or None, "reason": str}. None means the judge's
output could not be used, so it is left out of the averages and counted in the report.
Connection errors are NOT caught: if Ollama is down the run should stop, not record 50 failures.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.ingestion.embedding import get_dense_embeddings


class Statement(BaseModel):
    statement: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    statements: list[Statement]


class RelevanceCheck(BaseModel):
    questions: list[str]
    noncommittal: bool


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are checking an answer for faithfulness to its source context.

1. Break the answer into short, self-contained factual statements. Skip greetings and filler.
2. For each statement, decide whether the context below supports it. A statement is supported \
only if the context states it, or it follows directly from the context without any outside \
knowledge. Otherwise mark it unsupported, even if it is true in general.

Return every statement with its verdict.

Context:
{context}

Answer:
{answer}"""
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    """Read the answer below. Write 3 different questions that this answer would be a direct \
response to. Also set noncommittal to true if the answer is evasive, says it does not know, \
or says the information is not available. Otherwise set it to false.

Answer:
{answer}"""
)

CORRECTNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are grading a support assistant's answer against a reference.

Question: {question}
Reference answer: {reference}
Grading notes (the key facts the answer must contain): {notes}

Assistant's answer: {answer}

Grade the assistant's answer:
- correct: it states the key facts from the reference and the grading notes, and nothing in \
it contradicts them.
- partial: it contains some of the key facts but misses or garbles others.
- incorrect: it misses the key facts, contradicts the reference, or says the information is \
not available.
Give a one-sentence reason."""
)


def _reasoning() -> bool | str:
    value = settings.judge_reasoning
    return {"true": True, "false": False}.get(
        value.lower(), value
    )  # "low", "medium", "high" pass through


@lru_cache
def get_judge(schema: type[BaseModel]) -> Runnable:
    llm = ChatOllama(
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        seed=0,
        num_ctx=settings.judge_num_ctx,
        reasoning=_reasoning(),
    )
    return llm.with_structured_output(schema)


def _ask(
    prompt: ChatPromptTemplate, schema: type[BaseModel], llm: Runnable | None, **values
):
    """One judge call, retried once. Returns None if the output is unusable both times."""
    chain = prompt | (llm or get_judge(schema)).with_retry(stop_after_attempt=2)
    try:
        return chain.invoke(values)
    except (OutputParserException, ValidationError):
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def faithfulness(answer: str, contexts: list[str], llm: Runnable | None = None) -> dict:
    """Share of the answer's statements that the retrieved context supports (RAGAS definition)."""
    verdict = _ask(
        FAITHFULNESS_PROMPT,
        FaithfulnessVerdict,
        llm,
        context="\n\n---\n\n".join(contexts),
        answer=answer,
    )
    if verdict is None:
        return {"score": None, "reason": "judge output could not be parsed"}
    if not verdict.statements:
        return {"score": None, "reason": "no factual statements found in the answer"}
    unsupported = [s.statement for s in verdict.statements if not s.supported]
    score = 1 - len(unsupported) / len(verdict.statements)
    reason = (
        "all statements supported"
        if not unsupported
        else "unsupported: " + "; ".join(unsupported)
    )
    return {"score": score, "reason": reason}


def answer_relevance(
    question: str,
    answer: str,
    llm: Runnable | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict:
    """Does the answer address the question? (RAGAS definition.)

    The judge writes questions the answer would respond to. The score is their mean cosine
    similarity to the real question, and 0 if the answer is evasive.
    """
    check = _ask(RELEVANCE_PROMPT, RelevanceCheck, llm, answer=answer)
    if check is None:
        return {"score": None, "reason": "judge output could not be parsed"}
    if check.noncommittal or not check.questions:
        return {
            "score": 0.0,
            "reason": "the answer is evasive or says the information is not available",
        }
    embed = embed or get_dense_embeddings().embed_documents
    question_vector, *generated_vectors = embed([question] + check.questions)
    score = sum(_cosine(question_vector, v) for v in generated_vectors) / len(
        generated_vectors
    )
    return {
        "score": max(0.0, score),
        "reason": f"questions the answer responds to: {check.questions}",
    }


def answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    notes: str,
    llm: Runnable | None = None,
) -> dict:
    """Graded against ground_truth_answer and the key facts in notes: 1.0, 0.5 or 0.0."""
    verdict = _ask(
        CORRECTNESS_PROMPT,
        CorrectnessVerdict,
        llm,
        question=question,
        reference=ground_truth,
        notes=notes or "none",
        answer=answer,
    )
    if verdict is None:
        return {"score": None, "reason": "judge output could not be parsed"}
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[verdict.verdict]
    return {"score": score, "reason": verdict.reason}


def judge_all(item: dict, record: dict) -> dict:
    """All three LLM-judged metrics for one query (item = test set entry, record = pipeline result)."""
    return {
        "faithfulness": faithfulness(record["answer"], record["contexts"]),
        "answer_relevance": answer_relevance(item["query"], record["answer"]),
        "answer_correctness": answer_correctness(
            item["query"],
            record["answer"],
            item["ground_truth_answer"],
            item.get("notes", ""),
        ),
    }
