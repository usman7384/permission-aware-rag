from dataclasses import asdict, dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.common import qdrant_, redis_
from rag.common.config import Settings
from rag.common.ids import answer_point_id
from rag.common.log import log
from rag.common.text import extract_entities


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    doc_ids: list[str]


async def exact_cache(redis: Redis, cache_key: str) -> CachedAnswer | None:
    value = await redis_.get_answer(redis, cache_key)
    return CachedAnswer(**value) if value else None


async def semantic_cache(
    redis: Redis,
    qdrant: AsyncQdrantClient,
    settings: Settings,
    vector: list[float],
    fingerprint: str,
    question: str,
    request_id: str,
) -> CachedAnswer | None:
    """The answer cached for the nearest similar question with the same principals and
    entities; drops the point if its `ans:` key has expired."""
    nearest = await qdrant_.nearest_answer(qdrant, settings, vector, fingerprint)
    if nearest is None:
        return None
    point_id, similarity, payload = nearest
    if similarity < settings.semantic_cache_threshold:
        return None
    if payload["entities"] != extract_entities(question):
        return None
    cached = await exact_cache(redis, payload["cache_key"])
    if cached is None:
        await qdrant_.delete_answer(qdrant, settings, point_id)
        log("semantic_cache_expired", request_id=request_id, cache_key=payload["cache_key"])
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
        "doc_ids": cached.doc_ids,
    }
    await qdrant_.upsert_answer(qdrant, settings, str(answer_point_id(cache_key)), vector, payload)
