import argparse
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rag.api import pipeline
from rag.api.retrieve import RetrievalUnavailable
from rag.common import redis_
from rag.common.config import Settings
from rag.common.faults import FaultyLLM
from rag.evals.experiments import common
from rag.evals.score import TOP_K, Golden, mean, ndcg_at, percentile, recall_at, relevance_of

COMPONENTS = ("dense", "sparse", "reranker", "llm")
SCENARIOS: dict[str, dict[str, str]] = {
    "none": {},
    "dense down": {"dense": "down"},
    "dense slow": {"dense": "slow"},
    "sparse down": {"sparse": "down"},
    "sparse slow": {"sparse": "slow"},
    "both down": {"dense": "down", "sparse": "down"},
    "reranker down": {"reranker": "down"},
    "reranker slow": {"reranker": "slow"},
    "llm down": {"llm": "down"},
    "llm slow": {"llm": "slow"},
}


@dataclass(frozen=True)
class Trial:
    ranked_paths: list[str]
    degraded: list[str]
    kind: str
    ms: float


def answer_kind(outcome: pipeline.Outcome) -> str:
    """How the question was answered: by the LLM, by the extractive fallback, or not at all."""
    if outcome.abstained:
        return "abstained"
    return "extractive" if "llm" in outcome.degraded else "llm"


async def ask(deps: pipeline.Deps, question: Golden) -> Trial:
    """Steps 2 and 6 to 15 for one question, as `score.py` runs them; a retrieval failure is
    recorded as an error, as the API sends it."""
    start, request_id, timings = time.perf_counter(), str(uuid.uuid4()), {}
    async with deps.pg.connection() as conn:
        principals, _ = await pipeline.expand(conn, question.user)
    vector = await pipeline.embed(deps.embedder, question.question)
    try:
        parents, degraded = await pipeline.search(
            deps, principals, question.question, vector, request_id, timings
        )
    except RetrievalUnavailable:
        return Trial([], ["dense", "sparse"], "error", (time.perf_counter() - start) * 1000)
    outcome = pipeline.Outcome(degraded=degraded)
    async for _ in pipeline.answer(deps, parents, question.question, request_id, timings, outcome):
        pass
    ranked = list(dict.fromkeys(p.path for p in parents))
    ms = (time.perf_counter() - start) * 1000
    return Trial(ranked, outcome.degraded, answer_kind(outcome), ms)


def summarize(questions: list[Golden], trials: list[Trial]) -> dict[str, Any]:
    """Retrieval quality, latency and how the questions were answered, under one scenario."""
    pairs = list(zip(questions, trials, strict=True))
    kinds = Counter(t.kind for t in trials)
    return {
        "recall@10": mean([recall_at(t.ranked_paths, q.expected_paths, TOP_K) for q, t in pairs]),
        "ndcg@10": mean([ndcg_at(t.ranked_paths, relevance_of(q), TOP_K) for q, t in pairs]),
        "p50_ms": percentile([t.ms for t in trials], 50),
        "p95_ms": percentile([t.ms for t in trials], 95),
        **{kind: kinds[kind] / len(trials) for kind in ("llm", "extractive", "abstained", "error")},
        "degraded": ",".join(sorted({d for t in trials for d in t.degraded})) or "-",
    }


async def set_faults(deps: pipeline.Deps, faults: dict[str, str]) -> None:
    """Set the scenario's faults and clear every other component's."""
    for component in COMPONENTS:
        await redis_.set_fault(deps.redis, component, faults.get(component, "ok"))


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    if args.slow_ms is not None:
        settings = settings.model_copy(update={"fault_slow_ms": args.slow_ms})
    deps = await pipeline.open_deps(settings)
    fake = common.fake_llm_unless_configured(settings)
    deps.llm = FaultyLLM(fake, deps.redis, settings.fault_slow_ms) if fake else deps.llm
    runs = {}
    try:
        questions = await common.labelled_questions(deps.pg, args)
        for name in args.scenarios:
            await set_faults(deps, SCENARIOS[name])
            runs[name] = summarize(questions, [await ask(deps, q) for q in questions])
    finally:
        await set_faults(deps, {})
        await pipeline.close_deps(deps)
    llm_used = common.llm_name(settings)
    metrics = {"questions": len(questions), "llm": llm_used, "scenarios": runs}
    await common.save_run(settings, args, metrics)
    headers = ["scenario", "recall@10", "ndcg@10", "p50_ms", "p95_ms", "llm", "extractive"]
    headers += ["abstained", "error", "degraded"]
    rows = [[name, *run.values()] for name, run in runs.items()]
    footer = (
        f"\n{len(questions)} queries, {llm_used} LLM, FAULT_SLOW_MS={settings.fault_slow_ms}, "
        f"RETRIEVE_TIMEOUT_MS={settings.retrieve_timeout_ms}, "
        f"RERANK_TIMEOUT_MS={settings.rerank_timeout_ms}, LLM_TIMEOUT_S={settings.llm_timeout_s}"
    )
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    parser = common.parser("Answer quality and latency under each injected fault.", "failures")
    parser.set_defaults(limit=10)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--slow-ms", type=int, default=None, help="FAULT_SLOW_MS for this run")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
