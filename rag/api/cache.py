from dataclasses import asdict, dataclass
from typing import Any

from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.common import pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.ids import answer_point_id
from rag.common.keys import answer_key
from rag.common.log import log
from rag.common.permissions import readable_docs
from rag.common.text import extract_entities, negation_markers


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    doc_ids: list[str]
    doc_versions: dict[str, int]


@dataclass(frozen=True)
class CacheReader:
    """What a cache lookup needs: the stores and the caller's principals."""

    redis: Redis
    qdrant: AsyncQdrantClient
    pg: AsyncConnectionPool
    settings: Settings
    principals: list[str]
    request_id: str


async def still_valid(
    pool: AsyncConnectionPool, cached: CachedAnswer, principals: list[str]
) -> bool:
    """Every context document is live, at the cached version, and readable by `principals`."""
    docs = sorted(cached.doc_versions)
    async with pool.connection() as conn:
        versions = await pg.live_document_versions(conn, docs)
        readable = await readable_docs(conn, principals, docs)
    return bool(docs) and versions == cached.doc_versions and readable == set(docs)


async def drop_entry(reader: CacheReader, cache_key: str) -> None:
    """Delete a cached answer's `ans:` key and its `answer_cache` point."""
    await reader.redis.delete(answer_key(cache_key))
    point = str(answer_point_id(cache_key))
    await qdrant_.delete_answer(reader.qdrant, reader.settings, point)


async def exact_cache(reader: CacheReader, cache_key: str) -> CachedAnswer | None:
    """The answer cached under the key if it is still valid; an invalid one is dropped."""
    value = await redis_.get_answer(reader.redis, cache_key)
    if value is None:
        return None
    # An entry written before `doc_versions` existed cannot be checked, so it is dropped.
    cached = CachedAnswer(**value) if "doc_versions" in value else None
    if cached is not None and await still_valid(reader.pg, cached, reader.principals):
        return cached
    await drop_entry(reader, cache_key)
    log("cached_answer_invalid", request_id=reader.request_id, cache_key=cache_key)
    return None


def semantic_match(
    similarity: float,
    entities: list[str],
    negations: list[str] | None,
    question: str,
    threshold: float,
) -> bool:
    """Step 7's rule: similar enough, and the same entities and negation words as the cached
    question."""
    return (
        similarity >= threshold
        and entities == extract_entities(question)
        and negations == negation_markers(question)
    )


async def semantic_cache(
    reader: CacheReader, vector: list[float], fingerprint: str, question: str
) -> CachedAnswer | None:
    """The answer cached for the nearest similar question with the same principals, entities
    and negations; drops the point if its `ans:` key has expired or is no longer valid."""
    nearest = await qdrant_.nearest_answer(reader.qdrant, reader.settings, vector, fingerprint)
    if nearest is None:
        return None
    point_id, similarity, payload = nearest
    threshold = reader.settings.semantic_cache_threshold
    negations = payload.get("negations")
    if not semantic_match(similarity, payload["entities"], negations, question, threshold):
        return None
    cached = await exact_cache(reader, payload["cache_key"])
    if cached is None:
        await qdrant_.delete_answer(reader.qdrant, reader.settings, point_id)
        log("semantic_cache_expired", request_id=reader.request_id, cache_key=payload["cache_key"])
    return cached


async def write_caches(
    redis: Redis,
    qdrant: AsyncQdrantClient,
    settings: Settings,
    cache_key: str,
    fingerprint: str,
    question: str,
    vector: list[float],
    cached: CachedAnswer,
) -> None:
    """Section 8.4: the `ans:` key, the `doc2ans:` sets, then the `answer_cache` point."""
    await redis_.set_answer(redis, cache_key, asdict(cached), cached.doc_ids, settings.answer_ttl_s)
    payload = {
        "cache_key": cache_key,
        "fingerprint": fingerprint,
        "entities": extract_entities(question),
        "negations": negation_markers(question),
        "doc_ids": cached.doc_ids,
    }
    await qdrant_.upsert_answer(qdrant, settings, str(answer_point_id(cache_key)), vector, payload)
