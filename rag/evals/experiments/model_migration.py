import argparse
import asyncio
import time
import uuid
from collections import Counter
from typing import Any

from rag.api import pipeline
from rag.common import opensearch_, qdrant_
from rag.common.config import Settings
from rag.common.log import log
from rag.common.migrate import migrate
from rag.evals.experiments import common
from rag.evals.score import Golden, metrics_of, recall_depth, run_question

BATCH = 256


async def build_index(old: pipeline.Deps, new: pipeline.Deps) -> tuple[int, float]:
    """Re-embed every chunk of the old index with the new model into the new index version;
    returns the chunk count and the seconds taken."""
    start = time.perf_counter()
    await common.drop_indexes(new.settings)
    await migrate(new.settings, job_id="model-migration")
    chunks = await opensearch_.scroll_chunks(old.opensearch, old.settings)
    for first in range(0, len(chunks), BATCH):
        batch = chunks[first : first + BATCH]
        payloads = [{k: v for k, v in chunk.items() if k != "text"} for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        vectors = await asyncio.to_thread(new.embedder.embed_documents, texts)
        await qdrant_.upsert_chunks(new.qdrant, new.settings, payloads, vectors)
        await opensearch_.index_chunks(new.opensearch, new.settings, payloads, texts)
    return len(chunks), round(time.perf_counter() - start, 1)


async def serve(deps: pipeline.Deps, question: Golden) -> list[str]:
    """Steps 2 and 6 to 12 of one request; returns what was degraded."""
    async with deps.pg.connection() as conn:
        principals, _ = await pipeline.expand(conn, question.user)
    vector = await pipeline.embed(deps.embedder, question.question)
    request_id = str(uuid.uuid4())
    _, degraded = await pipeline.search(deps, principals, question.question, vector, request_id, {})
    return degraded


async def switch(
    old: pipeline.Deps, new: pipeline.Deps, questions: list[Golden], requests: int, concurrency: int
) -> Counter[str]:
    """Send requests concurrently and point new ones at the new index halfway through, while
    requests already running finish on the old one; counts served and failed per version."""
    active, counts, slots = [old], Counter[str](), asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with slots:
            if i == requests // 2:
                active[0] = new
            deps = active[0]
            version = deps.settings.index_version
            try:
                degraded = await serve(deps, questions[i % len(questions)])
            except Exception as exc:
                log("switch_request_failed", request_id=str(i), error=repr(exc))
                degraded = ["error"]
            counts[f"{version}_failed" if degraded else f"{version}_served"] += 1

    await asyncio.gather(*(one(i) for i in range(requests)))
    return counts


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    old_settings = recall_depth(settings)
    new_settings = old_settings.model_copy(
        update={
            "index_version": args.new_version,
            "embed_model": args.new_model,
            "embed_query_prefix": args.new_query_prefix,
        }
    )
    old = await pipeline.open_deps(old_settings)
    new = await pipeline.open_deps(new_settings)
    try:
        chunks, build_s = await build_index(old, new)
        questions = await common.labelled_questions(old.pg, args)
        scores = {}
        for deps in (old, new):
            results = [await run_question(deps, q, with_answers=False) for q in questions]
            scores[deps.settings.index_version] = metrics_of(questions, results, False)
        counts = await switch(old, new, questions, args.requests, args.concurrency)
    finally:
        await pipeline.close_deps(old)
        await pipeline.close_deps(new)
    rows, metrics = [], {"chunks": chunks, "build_s": build_s, "switch": dict(counts)}
    for deps in (old, new):
        s, m = deps.settings, scores[deps.settings.index_version]
        v = s.index_version
        metrics[v] = {"model": s.embed_model, **m}
        row: list[Any] = [v, s.embed_model, m["ndcg@10"], m["recall@100"], m["mrr"]]
        rows.append(row + [counts[f"{v}_served"], counts[f"{v}_failed"]])
    await common.save_run(settings, args, metrics)
    headers = ["index", "model", "ndcg@10", "recall@100", "mrr", "served", "failed"]
    footer = f"\n{new_settings.chunks_name}: {chunks} chunks re-embedded in {build_s} s"
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    parser = common.parser("Build, compare and switch to a second embedding model.", "migration")
    parser.add_argument("--new-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--new-query-prefix", default="", help="query prefix of the new model")
    parser.add_argument("--new-version", default="v2", help="INDEX_VERSION of the new index")
    parser.add_argument("--requests", type=int, default=40, help="requests during the switch")
    parser.add_argument("--concurrency", type=int, default=4, help="requests at once")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
