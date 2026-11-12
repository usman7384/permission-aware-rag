import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.api.auth import Caller
from rag.api.cache import CachedAnswer, CacheReader, exact_cache, semantic_cache, write_caches
from rag.api.generate import (
    ABSTAIN_TEXT,
    Context,
    Event,
    assemble,
    closing_events,
    generate,
    output_guard,
)
from rag.api.hydrate import ParentHit, hydrate
from rag.api.rerank import rerank
from rag.api.retrieve import RetrievalUnavailable, recheck, retrieve, rrf
from rag.api.rewrite import rewrite
from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.faults import FaultyLLM
from rag.common.keys import cache_key, fingerprint
from rag.common.log import log
from rag.common.models import (
    LLM,
    Embedder,
    InferenceUnavailable,
    Reranker,
    make_embedder,
    make_llm,
    make_reranker,
)
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
    inflight: dict[str, asyncio.Event] = field(default_factory=dict)


async def open_deps(settings: Settings) -> Deps:
    redis = redis_.connect(settings)
    return Deps(
        settings=settings,
        pg=await pg.open_pool(settings),
        redis=redis,
        qdrant=qdrant_.connect(settings),
        opensearch=opensearch_.connect(settings),
        embedder=make_embedder(settings),
        reranker=make_reranker(settings),
        llm=FaultyLLM(make_llm(settings), redis, settings.fault_slow_ms),
    )


async def close_deps(deps: Deps) -> None:
    await deps.pg.close()
    await deps.redis.aclose()
    await deps.qdrant.close()
    await deps.opensearch.close()


@dataclass
class Outcome:
    cache_tier: str = "none"
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    doc_versions: dict[str, int] = field(default_factory=dict)
    abstained: bool = False
    degraded: list[str] = field(default_factory=list)
    context: Context | None = None

    @property
    def cacheable(self) -> bool:
        fresh = self.cache_tier == "none" and not self.degraded
        return fresh and not self.abstained and bool(self.citations)


@contextmanager
def timed(timings: dict[str, float], step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[step] = round((time.perf_counter() - start) * 1000, 3)


async def timed_pieces(
    pieces: AsyncIterator[str], spent: dict[str, float], step: str
) -> AsyncIterator[str]:
    spent[step] = 0.0
    while True:
        start = time.perf_counter()
        try:
            piece = await anext(pieces)
        except StopAsyncIteration:
            return
        finally:
            spent[step] += (time.perf_counter() - start) * 1000
        yield piece


async def expand(conn: AsyncConnection, user: str) -> tuple[list[str], str]:
    principals = await expand_principals(conn, user)
    return principals, fingerprint(principals)


async def embed(embedder: Embedder, question: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_query, question)


@asynccontextmanager
async def coalesced(
    inflight: dict[str, asyncio.Event], key: str, wait_s: float, timings: dict[str, float]
) -> AsyncIterator[None]:
    running = inflight.get(key)
    if running is not None:
        with timed(timings, "coalesce"), suppress(TimeoutError):
            await asyncio.wait_for(running.wait(), wait_s)
        yield
        return
    inflight[key] = finished = asyncio.Event()
    try:
        yield
    finally:
        del inflight[key]
        finished.set()


async def lookup(
    deps: Deps,
    principals: list[str],
    key: str,
    fp: str,
    question: str,
    request_id: str,
    timings: dict[str, float],
) -> tuple[CachedAnswer | None, str, list[float] | None]:
    reader = CacheReader(deps.redis, deps.qdrant, deps.pg, deps.settings, principals, request_id)
    with timed(timings, "exact_cache"):
        cached = await exact_cache(reader, key)
    if cached is not None:
        return cached, "exact", None
    with timed(timings, "embed"):
        vector = await embed(deps.embedder, question)
    with timed(timings, "semantic_cache"):
        cached = await semantic_cache(reader, vector, fp, question)
    return cached, "semantic" if cached else "none", vector


async def search(
    deps: Deps,
    principals: list[str],
    question: str,
    vector: list[float],
    request_id: str,
    timings: dict[str, float],
) -> tuple[list[ParentHit], list[str]]:
    s = deps.settings
    with timed(timings, "retrieve"):
        branches, degraded = await retrieve(
            deps.qdrant, deps.opensearch, deps.redis, s, question, vector, principals, request_id
        )
    with timed(timings, "rrf"):
        fused = rrf(branches, s.rrf_k, s.fused_keep)
    with timed(timings, "recheck"):
        async with deps.pg.connection() as conn:
            allowed = await recheck(conn, fused, principals)
    with timed(timings, "rerank"):
        ranked, reranked = await rerank(
            deps.opensearch, deps.redis, deps.reranker, s, question, allowed, request_id
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
    reranked = "rerank" not in outcome.degraded
    with timed(timings, "assemble"):
        context = assemble(parents, reranked, deps.settings, deps.embedder.count_tokens, question)
    if context is None:
        outcome.answer, outcome.abstained = ABSTAIN_TEXT, True
        yield "token", {"text": ABSTAIN_TEXT}
        yield "citations", {"items": []}
        return
    spent: dict[str, float] = {}
    stream = generate(deps.llm, deps.settings, context, outcome.degraded, request_id)
    llm_pieces = timed_pieces(stream, spent, "generate")
    pieces = []
    async for piece in timed_pieces(output_guard(llm_pieces, context.parents), spent, "guard"):
        pieces.append(piece)
        yield "token", {"text": piece}
    outcome.answer = "".join(pieces)
    outcome.context = context
    outcome.doc_ids = sorted({p.doc_id for p in context.parents})
    outcome.doc_versions = {p.doc_id: p.version for p in context.parents}
    with timed(spent, "closing"):
        closing = closing_events(outcome.answer, context.parents)
    timings["generate"] = round(spent["generate"], 3)
    # Guard time excludes the time spent waiting on the LLM.
    timings["output_guard"] = round(spent["guard"] - spent["generate"] + spent["closing"], 3)
    for name, payload in closing:
        if name == "citations":
            outcome.citations = payload["items"]
        yield name, payload


def replay(cached: CachedAnswer, outcome: Outcome) -> list[Event]:
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
    async with pool.connection() as conn:
        with timed(timings, "persist"):
            async with conn.transaction():
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
        await pg.set_request_timings(conn, request_id, timings)


async def store_in_caches(
    deps: Deps,
    key: str,
    fp: str,
    question: str,
    vector: list[float],
    outcome: Outcome,
    request_id: str,
) -> None:
    cached = CachedAnswer(outcome.answer, outcome.citations, outcome.doc_ids, outcome.doc_versions)
    try:
        await write_caches(
            deps.redis, deps.qdrant, deps.settings, key, fp, question, vector, cached
        )
    except Exception as exc:
        log("cache_write_failed", request_id=request_id, error=repr(exc))


async def respond(
    deps: Deps,
    caller: Caller,
    principals: list[str],
    fp: str,
    session_id: str,
    question: str,
    rewritten: str | None,
    key: str,
    request_id: str,
    timings: dict[str, float],
    outcome: Outcome,
) -> AsyncIterator[Event]:
    query = rewritten or question
    try:
        cached, tier, vector = await lookup(deps, principals, key, fp, query, request_id, timings)
    except InferenceUnavailable as exc:
        log("ask_failed", request_id=request_id, code="embedding_unavailable", error=str(exc))
        yield "error", {"code": "embedding_unavailable", "message": str(exc)}
        return
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
    await persist(deps.pg, request_id, caller, session_id, question, rewritten, outcome, timings)
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
    # Caches are written only after the last event is sent.
    if outcome.cacheable and vector is not None:
        await store_in_caches(deps, key, fp, query, vector, outcome, request_id)


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
    key = cache_key(rewritten or question, fp, deps.settings.index_version)
    async with coalesced(deps.inflight, key, deps.settings.llm_timeout_s, timings):
        events = respond(
            deps,
            caller,
            principals,
            fp,
            session_id,
            question,
            rewritten,
            key,
            request_id,
            timings,
            outcome,
        )
        async for event in events:
            yield event
