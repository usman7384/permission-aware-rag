from rag.common import opensearch_, qdrant_
from rag.common.config import Settings
from rag.common.migrate import migrate


async def test_migrate_creates_test_collections_and_index(settings: Settings) -> None:
    await migrate(settings, job_id="test-migrate-again")
    qdrant = qdrant_.connect(settings)
    opensearch = opensearch_.connect(settings)
    try:
        chunks = await qdrant.get_collection("test_chunks_v1")
        cache = await qdrant.get_collection("test_answer_cache_v1")
        mapping = await opensearch.indices.get_mapping(index="test_chunks_v1")
    finally:
        await qdrant.close()
        await opensearch.close()
    assert chunks.config.params.vectors.size == 384
    assert set(chunks.payload_schema) == {"principals", "doc_id"}
    assert set(cache.payload_schema) == {"fingerprint", "doc_ids"}
    fields = mapping["test_chunks_v1"]["mappings"]["properties"]
    assert fields["principals"]["type"] == "keyword"
    assert fields["text"]["type"] == "text"
