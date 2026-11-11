import argparse
import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.api.generate import Context
from rag.api.pipeline import (
    Deps,
    Outcome,
    answer,
    close_deps,
    embed,
    expand,
    open_deps,
    search,
    timed,
)
from rag.common import pg
from rag.common.config import Settings
from rag.common.log import log
from rag.common.models import LLM
from rag.common.text import sentences
from rag.evals.benchmark import bench_name, bench_settings, read_qrels, read_queries

GOLDEN = Path(__file__).parent / "golden.jsonl"
JUDGE_SYSTEM = (Path(__file__).parents[2] / "prompts" / "judge_v1.txt").read_text(encoding="utf-8")
TOP_K = 10
BENCH_TOP_K = 100
# Store addresses and keys are not part of a run's configuration and may hold secrets.
NOT_CONFIG = {"llm_api_key", "pg_dsn", "redis_url", "qdrant_url", "opensearch_url"}


@dataclass(frozen=True)
class Golden:
    id: str
    user: str
    question: str
    expected_paths: list[str]
    # Graded relevance by path; empty means every expected path has grade 1.
    relevance: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    ranked_paths: list[str]
    judgements: list[int]
    timings: dict[str, float]


def load_golden(path: Path) -> list[Golden]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Golden(**json.loads(line)) for line in lines if line.strip()]


def recall_at(ranked: list[str], expected: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(expected)) / len(expected)


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    """1 / rank of the first expected path, or 0 if none is ranked."""
    return next((1 / rank for rank, p in enumerate(ranked, start=1) if p in expected), 0.0)


def ndcg_at(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain of the top k, with linear gains as in BEIR."""
    dcg = sum(relevance.get(p, 0) / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(i + 1) for i, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)) - 1, 0)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def latency(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """p50 and p95 in milliseconds of every stage that appears in the timings."""
    stages = dict.fromkeys(step for t in timings for step in t)
    return {
        step: {
            "p50": percentile([t[step] for t in timings if step in t], 50),
            "p95": percentile([t[step] for t in timings if step in t], 95),
        }
        for step in stages
    }


async def judge(llm: LLM, settings: Settings, context: Context, text: str) -> list[int]:
    """1 or 0 for each sentence of the answer: is it supported by the context?"""
    verdicts = []
    for sentence in (s.strip() for s in sentences(text)):
        if not sentence:
            continue
        async with asyncio.timeout(settings.llm_timeout_s):
            reply = await llm.complete(
                JUDGE_SYSTEM,
                f"{context.user}\n\nSentence: {sentence}",
                model=settings.llm_small_model or "",
                max_tokens=settings.judge_max_tokens,
            )
        verdicts.append(1 if reply.strip().startswith("1") else 0)
    return verdicts


async def run_question(deps: Deps, question: Golden, with_answers: bool = True) -> Result:
    """Steps 2 and 6 to 15 for one golden question as its user, with no cache and no persist;
    without answers, only steps 2 and 6 to 12."""
    request_id = str(uuid.uuid4())
    timings: dict[str, float] = {}
    async with deps.pg.connection() as conn:
        with timed(timings, "expand"):
            principals, _ = await expand(conn, question.user)
    with timed(timings, "embed"):
        vector = await embed(deps.embedder, question.question)
    parents, degraded = await search(
        deps, principals, question.question, vector, request_id, timings
    )
    outcome = Outcome(degraded=degraded)
    if with_answers:
        async for _ in answer(deps, parents, question.question, request_id, timings, outcome):
            pass
    judgements = []
    if outcome.context is not None:
        judgements = await judge(deps.llm, deps.settings, outcome.context, outcome.answer)
    ranked = list(dict.fromkeys(p.path for p in parents))
    return Result(ranked, judgements, timings)


def relevance_of(question: Golden) -> dict[str, int]:
    return question.relevance or dict.fromkeys(question.expected_paths, 1)


def metrics_of(
    questions: list[Golden], results: list[Result], with_answers: bool
) -> dict[str, Any]:
    pairs = list(zip(questions, results, strict=True))
    metrics: dict[str, Any] = {
        "questions": len(questions),
        "recall@10": mean([recall_at(r.ranked_paths, q.expected_paths, TOP_K) for q, r in pairs]),
        "recall@100": mean(
            [recall_at(r.ranked_paths, q.expected_paths, BENCH_TOP_K) for q, r in pairs]
        ),
        "ndcg@10": mean([ndcg_at(r.ranked_paths, relevance_of(q), TOP_K) for q, r in pairs]),
        "mrr": mean([reciprocal_rank(r.ranked_paths, q.expected_paths) for q, r in pairs]),
    }
    if with_answers:
        metrics["faithfulness"] = mean([v for r in results for v in r.judgements])
    metrics["latency_ms"] = latency([r.timings for r in results])
    return metrics


async def evaluate(
    deps: Deps, questions: list[Golden], name: str, with_answers: bool = True
) -> tuple[str, dict[str, Any]]:
    """Score the golden questions and store one `eval_runs` row; returns its id and metrics."""
    results = [await run_question(deps, q, with_answers) for q in questions]
    metrics = metrics_of(questions, results, with_answers)
    run_id = str(uuid.uuid4())
    config = deps.settings.model_dump(exclude=NOT_CONFIG)
    async with deps.pg.connection() as conn:
        await pg.insert_eval_run(conn, run_id, name, config, metrics)
    log("eval_done", job_id=run_id, name=name, metrics=metrics)
    return run_id, metrics


def table(metrics: dict[str, Any]) -> str:
    lines = [f"{'metric':<16}{'value':>10}"]
    names = ("recall@10", "recall@100", "ndcg@10", "mrr", "faithfulness")
    lines += [f"{m:<16}{metrics[m]:>10.3f}" for m in names if m in metrics]
    lines += ["", f"{'stage':<16}{'p50 ms':>10}{'p95 ms':>10}"]
    lines += [
        f"{step:<16}{v['p50']:>10.1f}{v['p95']:>10.1f}" for step, v in metrics["latency_ms"].items()
    ]
    return "\n".join(lines)


def benchmark_questions(
    folder: Path, split: str, user: str, paths: list[str], limit: int | None
) -> list[Golden]:
    """The labelled queries of a BEIR dataset, keeping relevant documents that are loaded."""
    path_of = {bench_name(path): path for path in paths}
    queries, questions = read_queries(folder), []
    for query_id, grades in read_qrels(folder, split).items():
        relevance = {path_of[d]: g for d, g in grades.items() if g > 0 and d in path_of}
        if relevance:
            questions.append(Golden(query_id, user, queries[query_id], list(relevance), relevance))
    return questions[:limit]


def recall_depth(settings: Settings) -> Settings:
    """The settings keeping enough candidates to measure recall@100."""
    keep = max(settings.fused_keep, BENCH_TOP_K)
    return settings.model_copy(update={"fused_keep": keep, "rerank_keep": BENCH_TOP_K})


def benchmark_settings(settings: Settings, args: argparse.Namespace) -> Settings:
    """The benchmark stores, keeping enough candidates to measure recall@100."""
    return recall_depth(bench_settings(settings, args.database, args.prefix))


async def run(args: argparse.Namespace) -> None:
    settings = Settings() if args.benchmark is None else benchmark_settings(Settings(), args)
    deps = await open_deps(settings)
    try:
        if args.benchmark is None:
            _, metrics = await evaluate(deps, load_golden(args.golden), args.name)
        else:
            async with deps.pg.connection() as conn:
                paths = await pg.live_document_paths(conn)
            folder, split, limit = args.benchmark, args.split, args.limit
            questions = benchmark_questions(folder, split, args.user, paths, limit)
            _, metrics = await evaluate(deps, questions, args.name, with_answers=False)
    finally:
        await close_deps(deps)
    print(table(metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the golden set and store the run.")
    parser.add_argument("--golden", type=Path, default=GOLDEN, help="golden set (JSON lines)")
    parser.add_argument("--name", default="golden", help="name of the eval_runs row")
    parser.add_argument("--benchmark", type=Path, help="score a loaded BEIR dataset folder instead")
    parser.add_argument("--split", default="test", help="benchmark qrels split")
    parser.add_argument("--user", default="user:bench", help="benchmark user")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N queries")
    parser.add_argument("--database", default="rag_bench", help="benchmark Postgres database")
    parser.add_argument("--prefix", default="bench_", help="benchmark Qdrant/OpenSearch prefix")
    asyncio.run(run(parser.parse_args()), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
