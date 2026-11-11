import argparse
import uuid

from rag.api import pipeline
from rag.api.hydrate import hydrate
from rag.api.rerank import rerank
from rag.api.retrieve import Candidate, dense, recheck, rrf, sparse
from rag.common.config import Settings
from rag.evals.experiments import common
from rag.evals.score import (
    BENCH_TOP_K,
    TOP_K,
    Golden,
    mean,
    ndcg_at,
    percentile,
    recall_at,
    recall_depth,
    relevance_of,
)

VARIANTS = ("sparse", "dense", "hybrid", "hybrid+rerank")


def variant_ms(timings: dict[str, float]) -> dict[str, float]:
    """Retrieval milliseconds of each variant; the pipeline runs the two branches concurrently."""
    dense_ms = timings["embed"] + timings["dense"]
    hybrid_ms = max(dense_ms, timings["sparse"])
    return {
        "sparse": timings["sparse"],
        "dense": dense_ms,
        "hybrid": hybrid_ms,
        "hybrid+rerank": hybrid_ms + timings["rerank"],
    }


async def ranked_by_variant(
    deps: pipeline.Deps, question: Golden
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Ranked document paths of each variant for one question, and the timings of each step."""
    s, timings, request_id = deps.settings, {}, str(uuid.uuid4())
    async with deps.pg.connection() as conn:
        principals, _ = await pipeline.expand(conn, question.user)
    with pipeline.timed(timings, "embed"):
        vector = await pipeline.embed(deps.embedder, question.question)
    with pipeline.timed(timings, "dense"):
        dense_hits = await dense(deps.qdrant, deps.redis, s, vector, principals, request_id)
    with pipeline.timed(timings, "sparse"):
        sparse_hits = await sparse(deps.opensearch, deps.redis, s, question.question, principals)
    fused = {"sparse": [sparse_hits], "dense": [dense_hits], "hybrid": [dense_hits, sparse_hits]}
    lists: dict[str, list[Candidate]] = {}
    async with deps.pg.connection() as conn:
        for name, branches in fused.items():
            lists[name] = await recheck(conn, rrf(branches, s.rrf_k, s.fused_keep), principals)
    with pipeline.timed(timings, "rerank"):
        reranked, ok = await rerank(
            deps.opensearch,
            deps.redis,
            deps.reranker,
            s,
            question.question,
            lists["hybrid"],
            request_id,
        )
    if not ok:
        raise RuntimeError("the reranker failed or timed out; raise --rerank-timeout-ms")
    lists["hybrid+rerank"] = reranked
    paths = {}
    for name, candidates in lists.items():
        parents = await hydrate(deps.pg, deps.redis, s.chunk_ttl_s, candidates[: s.rerank_keep])
        paths[name] = list(dict.fromkeys(p.path for p in parents))
    return paths, timings


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    deps = await pipeline.open_deps(recall_depth(settings))
    try:
        questions = await common.labelled_questions(deps.pg, args)
        runs = [await ranked_by_variant(deps, q) for q in questions]
    finally:
        await pipeline.close_deps(deps)
    metrics: dict[str, dict[str, float]] = {}
    for name in VARIANTS:
        pairs = [
            (q, paths[name], variant_ms(t)[name])
            for q, (paths, t) in zip(questions, runs, strict=True)
        ]
        metrics[name] = {
            "ndcg@10": mean([ndcg_at(r, relevance_of(q), TOP_K) for q, r, _ in pairs]),
            "recall@100": mean([recall_at(r, q.expected_paths, BENCH_TOP_K) for q, r, _ in pairs]),
            "p50_ms": percentile([ms for _, _, ms in pairs], 50),
        }
    await common.save_run(settings, args, {"questions": len(questions), "variants": metrics})
    rows = [[name, *metrics[name].values()] for name in VARIANTS]
    return common.table(["variant", "ndcg@10", "recall@100", "p50_ms"], rows)


def parser() -> argparse.ArgumentParser:
    return common.parser("nDCG@10 and recall@100 of each retrieval variant.", "ablation")


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
