from dataclasses import dataclass

from psycopg import AsyncConnection

from rag.common import pg
from rag.common.keys import api_key_hash


class ApiError(Exception):
    """An error returned before the response stream opens, as plain JSON."""

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


async def authenticate(conn: AsyncConnection, api_key: str | None) -> Caller:
    """Look up the user behind an `X-API-Key`; unknown or missing keys raise a 401."""
    key_hash = api_key_hash(api_key or "")
    row = await pg.get_api_key(conn, key_hash) if api_key else None
    if row is None:
        raise ApiError(401, "unauthorized", "unknown API key")
    user_id, is_admin = row
    return Caller(user_id=user_id, is_admin=is_admin, key_hash=key_hash)
