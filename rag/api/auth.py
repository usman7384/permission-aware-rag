from dataclasses import dataclass

from psycopg import AsyncConnection
from redis.asyncio import Redis

from rag.common import pg, redis_
from rag.common.config import Settings
from rag.common.keys import api_key_hash


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Caller:
    user_id: str
    is_admin: bool
    key_hash: str


async def authenticate(
    conn: AsyncConnection, redis: Redis, settings: Settings, api_key: str | None
) -> Caller:
    key_hash = api_key_hash(api_key or "")
    row = await pg.get_api_key(conn, key_hash) if api_key else None
    if row is None:
        raise ApiError(401, "unauthorized", "unknown API key")
    if await redis_.count_request(redis, key_hash, settings.rate_limit_ttl_s) > (
        settings.rate_limit_per_min
    ):
        raise ApiError(429, "rate_limited", "too many requests this minute")
    user_id, is_admin = row
    return Caller(user_id=user_id, is_admin=is_admin, key_hash=key_hash)
