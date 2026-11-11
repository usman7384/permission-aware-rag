from dataclasses import replace

import psycopg
import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from test_ingest import fetch, ingest
from test_tuple_changes import run_invalidator, stamped

from rag.common import pg, redis_
from rag.common.ids import doc_id
from rag.common.keys import CHANGES_GROUP, CHANGES_STREAM
from rag.common.permissions import change_tuple
from rag.worker import relay
from rag.worker.index import Deps

EXPENSES = "hr/expenses.md"


async def unpublished(deps: Deps) -> int:
    [(count,)] = await fetch(deps, "SELECT count(*) FROM outbox WHERE published_at IS NULL")
    return count


async def test_rolling_back_a_tuple_change_leaves_no_event(ingested: Deps) -> None:
    before = await fetch(ingested, "SELECT count(*) FROM outbox")
    async with ingested.pg.connection() as conn, conn.transaction():
        await change_tuple(conn, "add", "user:cal", "viewer", "folder:specs")
        cur = await conn.execute("SELECT count(*) FROM outbox")
        assert await cur.fetchone() != before[0]
        raise psycopg.Rollback()
    assert await fetch(ingested, "SELECT count(*) FROM outbox") == before
    assert await fetch(ingested, "SELECT * FROM tuples WHERE subject = 'user:cal'") == []


async def test_a_failed_document_commit_leaves_no_event(
    deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(*args: object) -> None:
        raise RuntimeError("parents write failed")

    monkeypatch.setattr(pg, "insert_parents", failing)
    await ingest(deps)
    assert await fetch(deps, "SELECT count(*) FROM documents") == [(0,)]
    assert await fetch(deps, "SELECT count(*) FROM outbox") == [(0,)]


async def test_a_change_committed_while_redis_is_down_is_published_later(ingested: Deps) -> None:
    await run_invalidator(ingested)
    async with ingested.pg.connection() as conn:
        await change_tuple(conn, "add", "user:cal", "viewer", f"doc:{doc_id(EXPENSES)}")
        await conn.commit()
    down = Redis.from_url("redis://127.0.0.1:1/15", socket_connect_timeout=1)
    try:
        with pytest.raises(RedisError):
            await relay.relay_once(replace(ingested, redis=down))
    finally:
        await down.aclose()
    assert await unpublished(ingested) == 1
    assert await relay.relay_once(ingested) == 1
    [(_, event)] = await ingested.redis.xrevrange(CHANGES_STREAM, count=1)
    assert event["type"] == "tuple_changed" and event["subject"] == "user:cal"
    await run_invalidator(ingested)
    assert await stamped(ingested, EXPENSES) == {("group:hr", "user:cal")}


async def test_a_relay_crash_after_xadd_gives_a_harmless_duplicate(
    ingested: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    await run_invalidator(ingested)
    async with ingested.pg.connection() as conn:
        await change_tuple(conn, "add", "user:cal", "viewer", f"doc:{doc_id(EXPENSES)}")
        await conn.commit()
    events = await ingested.redis.xlen(CHANGES_STREAM)

    async def crash(*args: object) -> None:
        raise RuntimeError("relay died before marking the events published")

    monkeypatch.setattr(pg, "mark_outbox_published", crash)
    with pytest.raises(RuntimeError):
        await relay.relay_once(ingested)
    monkeypatch.undo()
    assert await unpublished(ingested) == 1
    assert await relay.relay_once(ingested) == 1
    assert await ingested.redis.xlen(CHANGES_STREAM) == events + 2
    await run_invalidator(ingested)
    pending = await ingested.redis.xpending(CHANGES_STREAM, CHANGES_GROUP)
    assert pending["pending"] == 0
    assert await stamped(ingested, EXPENSES) == {("group:hr", "user:cal")}
    assert await unpublished(ingested) == 0
    assert await redis_.read_new(ingested.redis, CHANGES_STREAM, CHANGES_GROUP, "test", 10) == []
