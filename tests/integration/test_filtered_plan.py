import asyncio
import json
import random

import pytest
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff

from rag.api.retrieve import dense, plan_dense
from rag.common import qdrant_
from rag.common.config import Settings
from rag.worker.index import Deps

POINTS = 3000
FEW = 20


def unit_vector(rng: random.Random, dim: int) -> list[float]:
    vector = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def points(dim: int) -> list[tuple[dict, list[float]]]:
    rng = random.Random(0)
    found = []
    for i in range(POINTS):
        chunk = f"00000000-0000-0000-0000-{i:012d}"
        payload = {
            "chunk_id": chunk,
            "parent_id": chunk,
            "doc_id": chunk,
            "version": 1,
            "principals": ["user:few"] if i < FEW else ["group:broad"],
        }
        found.append((payload, unit_vector(rng, dim)))
    return found


async def build_collection(deps: Deps, s: Settings) -> None:
    await drop_collections(deps, s)
    await qdrant_.ensure_collections(deps.qdrant, s)
    await deps.qdrant.update_collection(
        s.chunks_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=10),
        hnsw_config=HnswConfigDiff(full_scan_threshold=10),
    )
    found = points(s.embed_dim)
    for first in range(0, len(found), 1000):
        batch = found[first : first + 1000]
        await qdrant_.upsert_chunks(deps.qdrant, s, [p for p, _ in batch], [v for _, v in batch])
    async with asyncio.timeout(600):
        while (await deps.qdrant.get_collection(s.chunks_name)).status != "green":
            await asyncio.sleep(0.2)


async def drop_collections(deps: Deps, s: Settings) -> None:
    await deps.qdrant.delete_collection(s.chunks_name)
    await deps.qdrant.delete_collection(s.answer_cache_name)


async def test_few_readable_chunks_are_scanned_exactly_and_many_use_hnsw(
    deps: Deps, capsys: pytest.CaptureFixture[str]
) -> None:
    s = deps.settings.model_copy(update={"index_version": "plan"})
    await build_collection(deps, s)
    try:
        info = await deps.qdrant.get_collection(s.chunks_name)
        assert info.indexed_vectors_count and info.indexed_vectors_count > POINTS // 2
        assert await plan_dense(deps.qdrant, s, ["user:few"]) == ("exact", FEW)
        assert await plan_dense(deps.qdrant, s, ["group:broad"]) == ("hnsw", POINTS - FEW)
        rng = random.Random(1)
        for _ in range(5):
            query = unit_vector(rng, s.embed_dim)
            capsys.readouterr()
            found = await dense(deps.qdrant, deps.redis, s, query, ["user:few"], "plan-test")
            truth = await qdrant_.search_chunks(
                deps.qdrant, s, query, ["user:few"], s.dense_top_k, exact=True
            )
            assert [c.chunk_id for c in found] == [p["chunk_id"] for p in truth]
            assert len(found) == FEW
            [plan] = [
                json.loads(line)
                for line in capsys.readouterr().out.splitlines()
                if '"dense_plan"' in line
            ]
            assert plan["plan"] == "exact" and plan["request_id"] == "plan-test"
    finally:
        await drop_collections(deps, s)
