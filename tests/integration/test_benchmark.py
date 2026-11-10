from pathlib import Path

import pytest

from rag.api import pipeline
from rag.common import pg, qdrant_
from rag.common.config import Settings
from rag.evals.benchmark import load, read_corpus
from rag.evals.score import benchmark_questions, evaluate
from rag.evals.synthetic import synthetic_acl
from rag.worker.index import Deps

SAMPLE = Path(__file__).parents[1] / "fixtures" / "beir" / "tiny"
USER = "user:bench"


async def load_sample(deps: Deps, folders: int = 1, selectivity: float = 1.0) -> dict:
    docs = read_corpus(SAMPLE)
    acl = synthetic_acl("tiny", len(docs), folders, selectivity, USER)
    return await load(deps, docs, acl, USER, batch_size=4)


async def count(deps: Deps, sql: str) -> int:
    async with deps.pg.connection() as conn:
        cur = await conn.execute(sql)
        return (await cur.fetchone())[0]


async def test_sample_is_loaded_in_batches_and_reloading_writes_nothing(deps: Deps) -> None:
    stats = await load_sample(deps)
    assert stats["documents"] == stats["indexed"] == 6
    assert await count(deps, "SELECT count(*) FROM documents WHERE path LIKE 'tiny/%'") == 6
    assert await count(deps, "SELECT count(*) FROM parents") == 6
    qdrant = await deps.qdrant.count(deps.settings.chunks_name)
    opensearch = await deps.opensearch.count(index=deps.settings.chunks_name)
    assert qdrant.count == opensearch["count"] == stats["chunks"] > 0
    again = await load_sample(deps)
    assert again["indexed"] == again["chunks"] == 0


async def test_benchmark_user_sees_only_its_folders_and_scores_on_the_labels(deps: Deps) -> None:
    await load_sample(deps, folders=2, selectivity=0.5)
    api = await pipeline.open_deps(deps.settings)
    try:
        async with api.pg.connection() as conn:
            paths = await pg.live_document_paths(conn)
            principals, _ = await pipeline.expand(conn, USER)
        questions = benchmark_questions(SAMPLE, "test", USER, paths, None)
        _, metrics = await evaluate(api, questions, "test-benchmark", with_answers=False)
    finally:
        await pipeline.close_deps(api)
    assert metrics["questions"] == 4 and "faithfulness" not in metrics
    # Documents are dealt round-robin over two folders; the user reads the first
    # (101, 103, 105), so it finds the labels of q1 and q2 but not of q3 (106) and q4 (102).
    assert metrics["recall@100"] == 0.5
    assert 0 < metrics["ndcg@10"] <= metrics["recall@100"]
    assert "group:tiny-f0000" in principals and "group:tiny-f0001" not in principals


async def test_loading_next_to_other_documents_is_refused(ingested: Deps) -> None:
    with pytest.raises(ValueError, match="other documents"):
        await load_sample(ingested)


@pytest.mark.parametrize("quantization", ["int8", "binary"])
async def test_quantization_and_on_disk_vectors_apply_only_when_set(
    settings: Settings, quantization: str
) -> None:
    quantized = settings.model_copy(
        update={
            "store_prefix": "test_quant_",
            "qdrant_quantization": quantization,
            "qdrant_vectors_on_disk": True,
        }
    )
    client = qdrant_.connect(quantized)
    try:
        default = (await client.get_collection(settings.chunks_name)).config.params.vectors
        await client.delete_collection(quantized.chunks_name)
        await qdrant_.ensure_collections(client, quantized)
        vectors = (await client.get_collection(quantized.chunks_name)).config.params.vectors
        cache = (await client.get_collection(quantized.answer_cache_name)).config.params.vectors
        await client.delete_collection(quantized.chunks_name)
        await client.delete_collection(quantized.answer_cache_name)
    finally:
        await client.close()
    assert default.quantization_config is None and default.memory is None
    kind = "scalar" if quantization == "int8" else "binary"
    assert getattr(vectors.quantization_config, kind) is not None
    assert vectors.memory == "cold"
    assert cache.quantization_config is None and cache.memory is None
