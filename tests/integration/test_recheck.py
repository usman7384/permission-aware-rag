from dataclasses import replace

from rag.api.retrieve import recheck, sparse
from rag.common.ids import doc_id
from rag.worker.index import Deps

ANA = ["group:eng", "group:staff", "user:ana"]
API_LIMITS = str(doc_id("specs/api-limits.md"))


async def test_chunk_behind_the_document_version_is_dropped(ingested: Deps) -> None:
    [current, *_] = await sparse(ingested.opensearch, ingested.settings, "ZEBRA-7", ANA)
    assert current.doc_id == API_LIMITS
    stale = replace(current, chunk_id="stale", version=current.version - 1)
    async with ingested.pg.connection() as conn:
        assert await recheck(conn, [stale, current], ANA) == [current]
        await conn.execute(
            "UPDATE documents SET version = version + 1 WHERE doc_id = %s", (API_LIMITS,)
        )
        assert await recheck(conn, [stale, current], ANA) == []


async def test_deleted_or_unreadable_documents_are_dropped(ingested: Deps) -> None:
    [current, *_] = await sparse(ingested.opensearch, ingested.settings, "ZEBRA-7", ANA)
    async with ingested.pg.connection() as conn:
        assert await recheck(conn, [current], ["group:hr", "group:staff", "user:ben"]) == [current]
        assert await recheck(conn, [current], ["user:cal"]) == []
        await conn.execute(
            "UPDATE documents SET deleted_at = now() WHERE doc_id = %s", (API_LIMITS,)
        )
        assert await recheck(conn, [current], ANA) == []
