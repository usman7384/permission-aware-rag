import json
import uuid

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from rag.common.config import Settings
from rag.common.keys import CHANGES_STREAM, DEAD_STREAM, JOBS_GROUP, JOBS_STREAM

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
