from redis.asyncio import Redis

from rag.common.config import Settings


def connect(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.store_timeout_s,
        socket_connect_timeout=settings.store_timeout_s,
    )


async def ping(client: Redis) -> None:
    await client.ping()
