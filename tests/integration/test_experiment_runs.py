from psycopg import AsyncConnection
from test_benchmark import SAMPLE, load_sample

from rag.common import redis_
from rag.evals.experiments import (
    ablation,
    cache_behaviour,
    chunk_size,
    common,
    contextual_chunking,
    cost,
    failure_injection,
    filtered_search,
    freshness,
    loadtest,
    model_migration,
    scale_memory,
)
from rag.worker.index import Deps


async def run_rows(deps: Deps, name: str) -> list[dict]:
    async with await AsyncConnection.connect(deps.settings.pg_dsn) as conn:
        cur = await conn.execute("SELECT metrics FROM eval_runs WHERE name = %s", (name,))
        return [row[0] for row in await cur.fetchall()]


async def run(deps: Deps, script, *argv: str) -> tuple[list[str], dict]:
    """Run the script on the tiny sample; returns its table lines and its one eval_runs row."""
    name = f"test-{script.__name__.rsplit('.', 1)[-1]}"
    args = script.parser().parse_args(["--benchmark", str(SAMPLE), "--name", name, *argv])
    text = await script.experiment(deps.settings, args)
    (metrics,) = await run_rows(deps, name)
    return text.splitlines(), metrics


async def test_ablation(deps: Deps) -> None:
    await load_sample(deps)
    lines, metrics = await run(deps, ablation)
    assert [line.split()[0] for line in lines[1:]] == list(ablation.VARIANTS)
    assert metrics["questions"] == 4
    assert metrics["variants"]["hybrid+rerank"]["recall@100"] == 1.0


async def test_scale_memory(deps: Deps) -> None:
    await load_sample(deps)
    args = ["--sizes", "3", "100", "--queries", "4"]
    lines, metrics = await run(deps, scale_memory, *args)
    assert len(lines) == 1 + 6 and [r["chunks"] for r in metrics["runs"]] == [3, 3, 3, 6, 6, 6]
    assert all(r["recall@10"] > 0 for r in metrics["runs"])
    collections = (await deps.qdrant.get_collections()).collections
    assert not [c.name for c in collections if "scale" in c.name]


async def test_filtered_search(deps: Deps) -> None:
    await load_sample(deps)
    args = ["--folders", "6", "--selectivities", "0.5", "0.17", "--queries", "4"]
    lines, metrics = await run(deps, filtered_search, *args, "--exact-max-points", "1")
    assert [round(r["selectivity"], 2) for r in metrics["runs"]] == [0.5, 0.17]
    assert [r["plan"] for r in metrics["runs"]] == ["hnsw", "exact"]
    assert metrics["runs"][1]["readable_chunks"] == 1
    assert metrics["runs"][1]["plan_recall"] == 1.0 and len(lines) == 3


async def test_model_migration(deps: Deps) -> None:
    await load_sample(deps)
    new = deps.settings.model_copy(update={"index_version": "v2"})
    try:
        lines, metrics = await run(deps, model_migration, "--requests", "6", "--concurrency", "2")
        assert await deps.qdrant.count(new.chunks_name) == await deps.qdrant.count(
            deps.settings.chunks_name
        )
    finally:
        await common.drop_indexes(new)
    assert metrics["switch"] == {"v1_served": 3, "v2_served": 3}
    assert metrics["v1"]["ndcg@10"] == metrics["v2"]["ndcg@10"]
    assert lines[0].split()[-2:] == ["served", "failed"] and len(lines) == 3 + 1


async def test_chunk_size(deps: Deps) -> None:
    lines, metrics = await run(deps, chunk_size, "--children", "8", "64", "--docs", "6")
    runs = metrics["runs"]
    assert sorted(runs) == ["c64", "c64p", "c8", "c8p"] and metrics["llm"] == "fake"
    assert runs["c8p"]["chunks"] > runs["c64p"]["chunks"] == 6
    assert all(r["questions"] == 4 and r["recall@10"] > 0 for r in runs.values())
    assert len(lines) == 1 + 4 + 1


async def test_contextual_chunking(deps: Deps) -> None:
    lines, metrics = await run(deps, contextual_chunking, "--docs", "6")
    assert metrics["plain"]["llm_calls"] == 0
    assert metrics["context"]["llm_calls"] == 6 and metrics["context"]["completion_tokens"] == 36
    assert metrics["context"]["prompt_tokens"] > 0 and len(lines) == 1 + 2 + 1


async def test_loadtest(deps: Deps) -> None:
    await load_sample(deps)
    args = ["--users", "1", "2", "--seconds", "2", "--stop-timeout-s", "5", "--port", "8799"]
    lines, metrics = await run(deps, loadtest, *args)
    for level in metrics["users"].values():
        assert level["request"]["n"] > 0 and level["request"]["failures"] == 0
        # The server stores every request Locust counted, and perhaps some it cut off.
        assert level["cache_hits"] + level["persist"]["n"] >= level["request"]["n"]
    assert metrics["users"]["1"]["retrieval"]["n"] > 0
    assert lines[0].split()[:2] == ["users", "stage"] and lines[1].split()[:2] == ["1", "request"]


async def test_loadtest_with_the_cache_prober(deps: Deps) -> None:
    await load_sample(deps)
    args = ["--users", "1", "--seconds", "2", "--stop-timeout-s", "5", "--port", "8798", "--hits"]
    lines, metrics = await run(deps, loadtest, *args)
    level = metrics["users"]["1"]
    assert level["hit"]["n"] > 0 and level["hit"]["failures"] == 0
    assert level["request"]["n"] > 0 and level["cache_hits"] >= level["hit"]["n"]
    assert lines[2].split()[:2] == ["1", "hit"]


async def test_cache_behaviour(deps: Deps) -> None:
    args = ["--intents", "4", "--requests", "40", "--thresholds", "0.9", "0.99"]
    lines, metrics = await run(deps, cache_behaviour, *args)
    loose, strict = metrics["thresholds"]["0.9"], metrics["thresholds"]["0.99"]
    assert loose["exact"] == strict["exact"] > 0
    assert loose["hit_rate"] >= strict["hit_rate"] and len(lines) == 1 + 2 + 1


async def test_freshness(deps: Deps) -> None:
    args = ["--docs", "6", "--trials", "1", "--redis-db", "14", "--poll-ms", "20"]
    lines, metrics = await run(deps, freshness, *args)
    assert metrics["searchable_s"][0] > 0 and metrics["evicted_s"][0] > 0
    assert len(lines) == 1 + 1 + 2 + 1
    collections = (await deps.qdrant.get_collections()).collections
    assert not [c.name for c in collections if "fresh" in c.name]


async def test_failure_injection(deps: Deps) -> None:
    await load_sample(deps)
    scenarios = ["none", "dense slow", "both down", "llm down"]
    args = ["--scenarios", *scenarios, "--slow-ms", "600"]
    lines, metrics = await run(deps, failure_injection, *args)
    runs = metrics["scenarios"]
    assert runs["none"]["llm"] == 1.0 and runs["none"]["degraded"] == "-"
    assert runs["dense slow"]["degraded"] == "dense" and runs["dense slow"]["recall@10"] > 0
    assert runs["both down"]["error"] == 1.0 and runs["llm down"]["extractive"] == 1.0
    faults = [await redis_.get_fault(deps.redis, c) for c in failure_injection.COMPONENTS]
    assert faults == [None] * 4
    assert len(lines) == 1 + 4 + 1


async def test_cost(deps: Deps) -> None:
    await load_sample(deps)
    lines, metrics = await run(deps, cost, "--top-k", "1", "3")
    one, three = metrics["top_k"]["1"], metrics["top_k"]["3"]
    assert three["tokens"]["answer_in"] > one["tokens"]["answer_in"] > 0
    assert one["tokens"]["rewrite_in"] > 0 and metrics["llm"] == "fake"
    usd = three["usd_per_1k"]
    assert usd["both"]["total"] < usd["plain"]["total"] and len(lines) == 1 + 2 + 1
