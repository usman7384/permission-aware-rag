import asyncio
from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.api.auth import Caller
from rag.api.hydrate import ParentHit, hydrate
from rag.api.rerank import rerank
from rag.api.retrieve import recheck, retrieve, rrf
from rag.common.config import Settings
from rag.common.keys import fingerprint
from rag.common.log import log
from rag.common.models import Embedder, Reranker
from rag.common.permissions import expand_principals


@dataclass
class Deps:
    settings: Settings
    pg: AsyncConnectionPool
    redis: Redis
    qdrant: AsyncQdrantClient
    opensearch: AsyncOpenSearch
    embedder: Embedder
    reranker: Reranker


async def expand(conn: AsyncConnection, user: str) -> tuple[list[str], str]:
    """The user's principals and their fingerprint."""
    principals = await expand_principals(conn, user)
    return principals, fingerprint(principals)


async def embed(embedder: Embedder, question: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_query, question)


async def search(
    deps: Deps, caller: Caller, question: str, request_id: str
) -> tuple[list[ParentHit], list[str]]:
    """Steps 2, 6 and 8 to 12: the parents the caller may read, best first, and `degraded`."""
    s = deps.settings
    async with deps.pg.connection() as conn:
        principals, _ = await expand(conn, caller.user_id)
    vector = await embed(deps.embedder, question)
    branches, degraded = await retrieve(
        deps.qdrant, deps.opensearch, s, question, vector, principals, request_id
    )
    fused = rrf(branches, s.rrf_k, s.fused_keep)
    async with deps.pg.connection() as conn:
        allowed = await recheck(conn, fused, principals)
    ranked, reranked = await rerank(
        deps.opensearch, deps.reranker, s, question, allowed, request_id
    )
    if not reranked:
        degraded.append("rerank")
    async with deps.pg.connection() as conn:
        parents = await hydrate(conn, ranked)
    log(
        "search_done",
        request_id=request_id,
        user=caller.user_id,
        parents=len(parents),
        degraded=degraded,
    )
    return parents, degraded
