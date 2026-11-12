import asyncio

from rag.common import pg, redis_
from rag.common.log import log
from rag.worker.index import Deps


async def relay_once(deps: Deps) -> int:
    async with deps.pg.connection() as conn, conn.transaction():
        rows = await pg.claim_outbox(conn, deps.settings.outbox_batch)
        for _, stream, payload in rows:
            await redis_.publish(deps.redis, stream, payload)
        if rows:
            await pg.mark_outbox_published(conn, [row_id for row_id, _, _ in rows])
    return len(rows)


async def relay_loop(deps: Deps) -> None:
    while True:
        try:
            published = await relay_once(deps)
        except Exception as exc:
            log("outbox_relay_failed", job_id="outbox", error=repr(exc))
            published = 0
        if published < deps.settings.outbox_batch:
            await asyncio.sleep(deps.settings.outbox_poll_ms / 1000)
