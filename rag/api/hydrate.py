from dataclasses import dataclass
from typing import Any

from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from rag.api.retrieve import Candidate
from rag.common import pg, redis_


@dataclass(frozen=True)
class ParentHit:
    parent_id: str
    doc_id: str
    version: int
    path: str
    heading: str | None
    text: str
    score: float


async def _from_postgres(pool: AsyncConnectionPool, parent_ids: list[str]) -> dict[str, Any]:
    async with pool.connection() as conn:
        rows = await pg.get_parents(conn, parent_ids)
    return {
        parent: {"doc_id": doc, "version": version, "path": path, "heading": heading, "text": text}
        for parent, doc, version, path, heading, text in rows
    }


async def hydrate(
    pool: AsyncConnectionPool, redis: Redis, chunk_ttl_s: int, candidates: list[Candidate]
) -> list[ParentHit]:
    best: dict[str, float] = {}
    for c in candidates:
        best[c.parent_id] = max(best.get(c.parent_id, c.score), c.score)
    parents = await redis_.get_parents(redis, list(best))
    missing = [parent for parent in best if parent not in parents]
    if missing:
        fetched = await _from_postgres(pool, missing)
        await redis_.set_parents(redis, fetched, chunk_ttl_s)
        parents |= fetched
    hits = [
        ParentHit(parent_id=parent, **parents[parent], score=score)
        for parent, score in best.items()
        if parent in parents
    ]
    return sorted(hits, key=lambda hit: -hit.score)
