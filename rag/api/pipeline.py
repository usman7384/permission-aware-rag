import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.api.auth import Caller
from rag.api.generate import ABSTAIN_TEXT, Event, assemble, closing_events, generate, output_guard
from rag.api.hydrate import ParentHit, hydrate
from rag.api.rerank import rerank
from rag.api.retrieve import RetrievalUnavailable, recheck, retrieve, rrf
from rag.common import pg
from rag.common.config import Settings
from rag.common.keys import fingerprint
from rag.common.log import log
from rag.common.models import LLM, Embedder, Reranker
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
    llm: LLM


@dataclass(frozen=True)
class Outcome:
    answer: str
    abstained: bool
    degraded: list[str]
    doc_ids: list[str]


@contextmanager
def timed(timings: dict[str, float], step: str) -> Iterator[None]:
    """Record the block's duration in milliseconds under `step`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[step] = round((time.perf_counter() - start) * 1000, 3)


async def expand(conn: AsyncConnection, user: str) -> tuple[list[str], str]:
    """The user's principals and their fingerprint."""
    principals = await expand_principals(conn, user)
    return principals, fingerprint(principals)


async def embed(embedder: Embedder, question: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_query, question)


async def search(
    deps: Deps, caller: Caller, question: str, request_id: str, timings: dict[str, float]
) -> tuple[list[ParentHit], list[str]]:
    """Steps 2, 6 and 8 to 12: the parents the caller may read, best first, and `degraded`."""
    s = deps.settings
    with timed(timings, "expand"):
        async with deps.pg.connection() as conn:
            principals, _ = await expand(conn, caller.user_id)
    with timed(timings, "embed"):
        vector = await embed(deps.embedder, question)
    with timed(timings, "retrieve"):
        branches, degraded = await retrieve(
            deps.qdrant, deps.opensearch, s, question, vector, principals, request_id
        )
    with timed(timings, "rrf"):
        fused = rrf(branches, s.rrf_k, s.fused_keep)
    with timed(timings, "recheck"):
        async with deps.pg.connection() as conn:
            allowed = await recheck(conn, fused, principals)
    with timed(timings, "rerank"):
        ranked, reranked = await rerank(
            deps.opensearch, deps.reranker, s, question, allowed, request_id
        )
    if not reranked:
        degraded.append("rerank")
    with timed(timings, "hydrate"):
        async with deps.pg.connection() as conn:
            parents = await hydrate(conn, ranked)
    return parents, degraded


async def persist(
    pool: AsyncConnectionPool,
    request_id: str,
    caller: Caller,
    session_id: str,
    question: str,
    outcome: Outcome,
    timings: dict[str, float],
) -> None:
    """Store the user and assistant turns and the request row in one transaction."""
    async with pool.connection() as conn, conn.transaction():
        await pg.insert_turn(conn, session_id, caller.user_id, "user", question)
        await pg.insert_turn(conn, session_id, caller.user_id, "assistant", outcome.answer)
        await pg.insert_request(
            conn,
            request_id,
            caller.user_id,
            session_id,
            question,
            "none",
            outcome.abstained,
            outcome.degraded,
            outcome.doc_ids,
            timings,
            outcome.answer,
        )


async def ask(
    deps: Deps,
    caller: Caller,
    session_id: str,
    question: str,
    request_id: str,
    timings: dict[str, float],
) -> AsyncIterator[Event]:
    """Steps 2 and 6 to 16 for an authenticated caller, as server-sent events."""
    try:
        parents, degraded = await search(deps, caller, question, request_id, timings)
    except RetrievalUnavailable as exc:
        log("ask_failed", request_id=request_id, code="retrieval_unavailable", error=str(exc))
        yield "error", {"code": "retrieval_unavailable", "message": str(exc)}
        return
    with timed(timings, "assemble"):
        context = assemble(
            parents, "rerank" not in degraded, deps.settings, deps.embedder.count_tokens, question
        )
    if context is None:
        text = ABSTAIN_TEXT
        yield "token", {"text": text}
        yield "citations", {"items": []}
    else:
        pieces = []
        with timed(timings, "generate"):
            stream = generate(deps.llm, deps.settings, context, degraded, request_id)
            async for piece in output_guard(stream, context.parents):
                pieces.append(piece)
                yield "token", {"text": piece}
        text = "".join(pieces)
        for event in closing_events(text, context.parents):
            yield event
    doc_ids = sorted({p.doc_id for p in context.parents}) if context else []
    outcome = Outcome(text, context is None, degraded, doc_ids)
    with timed(timings, "persist"):
        await persist(deps.pg, request_id, caller, session_id, question, outcome, timings)
    log(
        "ask_done",
        request_id=request_id,
        user=caller.user_id,
        abstained=outcome.abstained,
        degraded=degraded,
        timings=timings,
    )
    yield (
        "done",
        {
            "request_id": request_id,
            "cache_tier": "none",
            "degraded": degraded,
            "abstained": outcome.abstained,
        },
    )
