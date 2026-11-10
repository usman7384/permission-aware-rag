import json
import time
import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from rag.common.config import Settings
from rag.common.keys import (
    CHANGES_STREAM,
    DEAD_STREAM,
    JOBS_GROUP,
    JOBS_STREAM,
    answer_key,
    chunk_key,
    doc_answers_key,
    rate_limit_key,
)

Message = tuple[str, dict[str, str]]


def connect(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.store_timeout_s,
        socket_connect_timeout=settings.store_timeout_s,
    )


async def ping(client: Redis) -> None:
    await client.ping()


async def ensure_group(client: Redis, stream: str, group: str) -> None:
    try:
        await client.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def enqueue_job(client: Redis, op: str, path: str) -> str:
    job_id = uuid.uuid4().hex
    await client.xadd(JOBS_STREAM, {"job_id": job_id, "op": op, "path": path})
    return job_id


async def read_new(
    client: Redis, stream: str, group: str, consumer: str, block_ms: int
) -> list[Message]:
    """New messages for this consumer; waits up to `block_ms` for one to arrive."""
    reply = await client.xreadgroup(group, consumer, {stream: ">"}, count=1, block=block_ms)
    return [message for _, messages in reply for message in messages]


async def claim_idle(
    client: Redis, stream: str, group: str, consumer: str, min_idle_ms: int
) -> list[Message]:
    """Take over pending messages that no consumer has touched for `min_idle_ms`."""
    _, messages, _ = await client.xautoclaim(stream, group, consumer, min_idle_time=min_idle_ms)
    return [(message_id, fields) for message_id, fields in messages if fields]


async def deliveries(client: Redis, message_id: str) -> int:
    pending = await client.xpending_range(
        JOBS_STREAM, JOBS_GROUP, min=message_id, max=message_id, count=1
    )
    return pending[0]["times_delivered"] if pending else 0


async def ack(client: Redis, stream: str, group: str, message_id: str) -> None:
    await client.xack(stream, group, message_id)


async def dead_letter(
    client: Redis, message_id: str, fields: dict[str, str], error: str, delivered: int
) -> None:
    """Copy the job to `ingest:dead` and acknowledge it, atomically."""
    async with client.pipeline(transaction=True) as pipe:
        pipe.xadd(DEAD_STREAM, {**fields, "error": error, "deliveries": str(delivered)})
        pipe.xack(JOBS_STREAM, JOBS_GROUP, message_id)
        await pipe.execute()


async def emit_change(client: Redis, event_type: str, doc_ids: list[str]) -> None:
    await client.xadd(CHANGES_STREAM, {"type": event_type, "doc_ids": json.dumps(doc_ids)})


async def emit_tuple_change(client: Redis, subject: str, relation: str, obj: str) -> None:
    await client.xadd(
        CHANGES_STREAM,
        {"type": "tuple_changed", "subject": subject, "relation": relation, "object": obj},
    )


async def count_request(client: Redis, key_hash: str, ttl_s: int) -> int:
    """Increment the key's counter for the current minute and return it."""
    key = rate_limit_key(key_hash, int(time.time()) // 60)
    async with client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, ttl_s)
        count, _ = await pipe.execute()
    return count


async def get_answer(client: Redis, cache_key: str) -> dict[str, Any] | None:
    value = await client.get(answer_key(cache_key))
    return json.loads(value) if value else None


async def set_answer(
    client: Redis, cache_key: str, answer: dict[str, Any], doc_ids: list[str], ttl_s: int
) -> None:
    """Write `ans:` and add the key to each document's `doc2ans:` set, refreshing its TTL."""
    async with client.pipeline(transaction=True) as pipe:
        pipe.set(answer_key(cache_key), json.dumps(answer), ex=ttl_s)
        for doc in doc_ids:
            pipe.sadd(doc_answers_key(doc), cache_key)
            pipe.expire(doc_answers_key(doc), ttl_s)
        await pipe.execute()


async def purge_answers(client: Redis, doc_ids: list[str]) -> list[str]:
    """Delete the `ans:` keys listed in the documents' `doc2ans:` sets, then the sets."""
    cache_keys: set[str] = set()
    for doc in doc_ids:
        cache_keys |= await client.smembers(doc_answers_key(doc))
    keys = [answer_key(key) for key in cache_keys] + [doc_answers_key(doc) for doc in doc_ids]
    await client.delete(*keys)
    return sorted(cache_keys)


async def get_parents(client: Redis, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
    """The cached `chunk:` values of the listed parents that are present."""
    if not parent_ids:
        return {}
    values = await client.mget([chunk_key(parent) for parent in parent_ids])
    return {
        parent: json.loads(value)
        for parent, value in zip(parent_ids, values, strict=True)
        if value is not None
    }


async def set_parents(client: Redis, parents: dict[str, dict[str, Any]], ttl_s: int) -> None:
    async with client.pipeline(transaction=False) as pipe:
        for parent, value in parents.items():
            pipe.set(chunk_key(parent), json.dumps(value), ex=ttl_s)
        await pipe.execute()


async def delete_parents_of(client: Redis, doc_ids: list[str]) -> int:
    """Delete every `chunk:` key whose parent belongs to one of the documents; returns how many."""
    keys = [key async for key in client.scan_iter(match=chunk_key("*"))]
    values = await client.mget(keys) if keys else []
    stale = [
        key
        for key, value in zip(keys, values, strict=True)
        if value is not None and json.loads(value)["doc_id"] in doc_ids
    ]
    if stale:
        await client.delete(*stale)
    return len(stale)
