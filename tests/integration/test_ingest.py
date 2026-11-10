from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.common import redis_
from rag.common.ids import doc_id
from rag.common.keys import CHANGES_STREAM, JOBS_GROUP, JOBS_STREAM
from rag.worker.chunk import chunk
from rag.worker.index import Deps
from rag.worker.main import consume_once
from rag.worker.parse import parse
from rag.worker.watch import corpus_files, reconcile

API_LIMITS = "specs/api-limits.md"


async def ingest(deps: Deps) -> None:
    """Reconcile the corpus and process jobs until the stream is drained."""
    await redis_.ensure_group(deps.redis, JOBS_STREAM, JOBS_GROUP)
    await reconcile(deps.settings, deps.pg, deps.redis)
    while await consume_once(deps, "test"):
        pass


async def qdrant_versions(deps: Deps, doc: str | None = None) -> list[int]:
    match = [FieldCondition(key="doc_id", match=MatchValue(value=doc))] if doc else []
    condition = Filter(must=match)
    points, _ = await deps.qdrant.scroll(
        deps.settings.chunks_name, scroll_filter=condition, limit=10_000
    )
    return [point.payload["version"] for point in points]


async def opensearch_versions(deps: Deps, doc: str | None = None) -> list[int]:
    query = {"term": {"doc_id": doc}} if doc else {"match_all": {}}
    response = await deps.opensearch.search(
        index=deps.settings.chunks_name, body={"query": query, "size": 10_000}
    )
    return [hit["_source"]["version"] for hit in response["hits"]["hits"]]


async def fetch(deps: Deps, sql: str, *params: object) -> list[tuple]:
    async with deps.pg.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


def expected_chunks(deps: Deps) -> int:
    s, corpus = deps.settings, Path(deps.settings.corpus_dir)
    return sum(
        len(parent.children)
        for path in corpus_files(corpus)
        for parent in chunk(
            parse(path, (corpus / path).read_bytes()),
            s.parent_tokens,
            s.child_tokens,
            deps.embedder.count_tokens,
        )
    )


async def test_fixture_corpus_yields_12_documents_and_matching_chunks(deps: Deps) -> None:
    await ingest(deps)
    docs = await fetch(deps, "SELECT count(*) FROM documents WHERE deleted_at IS NULL")
    assert docs == [(12,)]
    expected = expected_chunks(deps)
    assert len(await qdrant_versions(deps)) == expected
    assert len(await opensearch_versions(deps)) == expected
    points, _ = await deps.qdrant.scroll(deps.settings.chunks_name, limit=10_000)
    readers = {
        p.payload["principals"] == ["group:eng", "user:ben"]
        for p in points
        if p.payload["doc_id"] == str(doc_id(API_LIMITS))
    }
    assert readers == {True}


async def test_reingesting_unchanged_files_writes_nothing(deps: Deps) -> None:
    await ingest(deps)
    before = await fetch(deps, "SELECT doc_id, version, updated_at FROM documents ORDER BY 1")
    events = await deps.redis.xlen(CHANGES_STREAM)
    await ingest(deps)
    assert await fetch(deps, "SELECT doc_id, version, updated_at FROM documents ORDER BY 1") == (
        before
    )
    assert await deps.redis.xlen(CHANGES_STREAM) == events


async def test_changed_file_bumps_version_and_leaves_no_old_chunks(deps: Deps) -> None:
    await ingest(deps)
    doc = str(doc_id(API_LIMITS))
    file = Path(deps.settings.corpus_dir) / API_LIMITS
    file.write_text(file.read_text(encoding="utf-8") + "\n## Changes\n\nNew text.\n", "utf-8")
    await ingest(deps)
    assert await fetch(deps, "SELECT version FROM documents WHERE doc_id = %s", doc) == [(2,)]
    assert set(await qdrant_versions(deps, doc)) == {2}
    assert set(await opensearch_versions(deps, doc)) == {2}
    parents = await fetch(deps, "SELECT DISTINCT version FROM parents WHERE doc_id = %s", doc)
    assert parents == [(2,)]
    assert len(await qdrant_versions(deps, doc)) == len(await opensearch_versions(deps, doc))


async def test_deleted_file_removes_chunks_parents_and_sets_deleted_at(deps: Deps) -> None:
    await ingest(deps)
    doc = str(doc_id(API_LIMITS))
    (Path(deps.settings.corpus_dir) / API_LIMITS).unlink()
    await ingest(deps)
    deleted = await fetch(
        deps, "SELECT deleted_at IS NOT NULL FROM documents WHERE doc_id = %s", doc
    )
    assert deleted == [(True,)]
    assert await fetch(deps, "SELECT count(*) FROM parents WHERE doc_id = %s", doc) == [(0,)]
    assert await qdrant_versions(deps, doc) == []
    assert await opensearch_versions(deps, doc) == []
    parent_tuples = await fetch(deps, "SELECT * FROM tuples WHERE subject = %s", f"doc:{doc}")
    assert parent_tuples == []
    [(_, last_event)] = await deps.redis.xrevrange(CHANGES_STREAM, count=1)
    assert last_event == {"type": "doc_deleted", "doc_ids": f'["{doc}"]'}
