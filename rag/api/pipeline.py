import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.api.auth import Caller
from rag.api.cache import CachedAnswer, exact_cache, semantic_cache, write_caches
from rag.api.generate import ABSTAIN_TEXT, Event, assemble, closing_events, generate, output_guard
from rag.api.hydrate import ParentHit, hydrate
from rag.api.rerank import rerank
from rag.api.retrieve import RetrievalUnavailable, recheck, retrieve, rrf
from rag.api.rewrite import rewrite
from rag.common import pg
from rag.common.config import Settings
from rag.common.keys import cache_key, fingerprint
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


@dataclass
class Outcome:
    cache_tier: str = "none"
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    abstained: bool = False
    degraded: list[str] = field(default_factory=list)

    @property
    def cacheable(self) -> bool:
        """Section 8.4: a fresh, undegraded, cited answer."""
        fresh = self.cache_tier == "none" and not self.degraded
        return fresh and not self.abstained and bool(self.citations)


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


async def lookup(
    deps: Deps,
    key: str,
    fp: str,
    question: str,
    request_id: str,
    timings: dict[str, float],
) -> tuple[CachedAnswer | None, str, list[float] | None]:
    """Steps 5 to 7: a cached answer and its tier, and the query vector if it was computed."""
    with timed(timings, "exact_cache"):
        cached = await exact_cache(deps.redis, key)
    if cached is not None:
        return cached, "exact", None
    with timed(timings, "embed"):
        vector = await embed(deps.embedder, question)
    with timed(timings, "semantic_cache"):
        cached = await semantic_cache(
            deps.redis, deps.qdrant, deps.settings, vector, fp, question, request_id
        )
    return cached, "semantic" if cached else "none", vector


async def search(
    deps: Deps,
    principals: list[str],
    question: str,
    vector: list[float],
    request_id: str,
    timings: dict[str, float],
) -> tuple[list[ParentHit], list[str]]:
    """Steps 8 to 12: the parents the principals may read, best first, and `degraded`."""
    s = deps.settings
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
        parents = await hydrate(deps.pg, deps.redis, s.chunk_ttl_s, ranked)
    return parents, degraded


async def answer(
    deps: Deps,
    parents: list[ParentHit],
    question: str,
    request_id: str,
    timings: dict[str, float],
    outcome: Outcome,
) -> AsyncIterator[Event]:
    """Steps 13 to 15 as events; fills `outcome` with the answer, citations and documents."""
    reranked = "rerank" not in outcome.degraded
    with timed(timings, "assemble"):
        context = assemble(parents, reranked, deps.settings, deps.embedder.count_tokens, question)
    if context is None:
        outcome.answer, outcome.abstained = ABSTAIN_TEXT, True
        yield "token", {"text": ABSTAIN_TEXT}
        yield "citations", {"items": []}
        return
    pieces = []
    with timed(timings, "generate"):
        stream = generate(deps.llm, deps.settings, context, outcome.degraded, request_id)
        async for piece in output_guard(stream, context.parents):
            pieces.append(piece)
            yield "token", {"text": piece}
    outcome.answer = "".join(pieces)
    outcome.doc_ids = sorted({p.doc_id for p in context.parents})
    for name, payload in closing_events(outcome.answer, context.parents):
        if name == "citations":
            outcome.citations = payload["items"]
        yield name, payload


def replay(cached: CachedAnswer, outcome: Outcome) -> list[Event]:
    """The events of a cached answer; fills `outcome` from it."""
    outcome.answer = cached.answer
    outcome.citations = cached.citations
    outcome.doc_ids = cached.doc_ids
    return [("token", {"text": cached.answer}), ("citations", {"items": cached.citations})]


async def persist(
    pool: AsyncConnectionPool,
    request_id: str,
    caller: Caller,
    session_id: str,
    question: str,
    rewritten: str | None,
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
            rewritten,
            outcome.cache_tier,
            outcome.abstained,
            outcome.degraded,
            outcome.doc_ids,
            timings,
            outcome.answer,
        )


async def store_in_caches(
    deps: Deps,
    key: str,
    fp: str,
    question: str,
    vector: list[float],
    outcome: Outcome,
    request_id: str,
) -> None:
    """Write a cacheable answer to the caches; a failure is only logged, the answer is sent."""
    cached = CachedAnswer(outcome.answer, outcome.citations, outcome.doc_ids)
    try:
        await write_caches(
            deps.redis, deps.qdrant, deps.settings, key, fp, question, vector, cached
        )
    except Exception as exc:
        log("cache_write_failed", request_id=request_id, error=repr(exc))


async def ask(
    deps: Deps,
    caller: Caller,
    principals: list[str],
    fp: str,
    session_id: str,
    question: str,
    request_id: str,
    timings: dict[str, float],
) -> AsyncIterator[Event]:
    """Steps 4 to 16 for an authenticated, guarded question, as server-sent events."""
    outcome = Outcome()
    with timed(timings, "rewrite"):
        rewritten = await rewrite(
            deps.pg,
            deps.llm,
            deps.settings,
            caller.user_id,
            session_id,
            question,
            outcome.degraded,
            request_id,
        )
    query = rewritten or question
    key = cache_key(query, fp, deps.settings.index_version)
    cached, tier, vector = await lookup(deps, key, fp, query, request_id, timings)
    outcome.cache_tier = tier
    if cached is not None:
        for event in replay(cached, outcome):
            yield event
    elif vector is not None:
        try:
            parents, failed = await search(deps, principals, query, vector, request_id, timings)
        except RetrievalUnavailable as exc:
            log("ask_failed", request_id=request_id, code="retrieval_unavailable", error=str(exc))
            yield "error", {"code": "retrieval_unavailable", "message": str(exc)}
            return
        outcome.degraded += failed
        async for event in answer(deps, parents, query, request_id, timings, outcome):
            yield event
    with timed(timings, "persist"):
        await persist(
            deps.pg, request_id, caller, session_id, question, rewritten, outcome, timings
        )
    log(
        "ask_done",
        request_id=request_id,
        user=caller.user_id,
        cache_tier=tier,
        abstained=outcome.abstained,
        degraded=outcome.degraded,
        timings=timings,
    )
    yield (
        "done",
        {
            "request_id": request_id,
            "cache_tier": tier,
            "degraded": outcome.degraded,
            "abstained": outcome.abstained,
        },
    )
    # Step 16: the caches are written after the last event has been sent.
    if outcome.cacheable and vector is not None:
        await store_in_caches(deps, key, fp, query, vector, outcome, request_id)
