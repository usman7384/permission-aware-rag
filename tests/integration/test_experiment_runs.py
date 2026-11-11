from psycopg import AsyncConnection
from test_benchmark import SAMPLE, load_sample

from rag.evals.experiments import (
    ablation,
    chunk_size,
    common,
    contextual_chunking,
    filtered_search,
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
    assert [r["fallback"] for r in metrics["runs"]] == ["ann", "exact"]
    assert metrics["runs"][1]["readable_chunks"] == 1
    assert metrics["runs"][1]["fallback_recall"] == 1.0 and len(lines) == 3


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
