import json
import random

import pytest

from rag.api.retrieve import dense, plan_dense
from rag.common import qdrant_
from rag.evals.experiments import common
from rag.worker.index import Deps

POINTS = 3000
FEW = 20


def unit_vector(rng: random.Random, dim: int) -> list[float]:
    vector = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector]


def points(dim: int) -> common.Points:
    """Random chunks: the first `FEW` readable by `user:few`, the rest by `group:broad`."""
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


async def test_few_readable_chunks_are_scanned_exactly_and_many_use_hnsw(
    deps: Deps, capsys: pytest.CaptureFixture[str]
) -> None:
    s = deps.settings.model_copy(update={"index_version": "plan"})
    await common.build_collection(deps.qdrant, s, points(s.embed_dim), threshold_kb=10)
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
        await common.drop_collections(deps.qdrant, s)
