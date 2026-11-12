import asyncio

from psycopg import AsyncConnection

from rag.common import pg
from rag.common.config import Settings
from rag.common.ids import doc_id
from rag.common.keys import api_key_hash
from rag.common.log import log

API_KEYS = [
    ("dev-ana", "user:ana", False),
    ("dev-ben", "user:ben", False),
    ("dev-cal", "user:cal", False),
    ("dev-admin", "user:admin", True),
]

TUPLES = [
    ("user:ana", "member", "group:eng"),
    ("user:ben", "member", "group:hr"),
    ("group:eng", "member", "group:staff"),
    ("group:hr", "member", "group:staff"),
    ("group:eng", "viewer", "folder:specs"),
    ("group:hr", "viewer", "folder:hr"),
    ("group:staff", "viewer", "folder:public"),
    ("user:ben", "viewer", f"doc:{doc_id('specs/api-limits.md')}"),
]


async def seed(conn: AsyncConnection) -> None:
    async with conn.transaction():
        for key, user_id, is_admin in API_KEYS:
            await pg.upsert_api_key(conn, api_key_hash(key), user_id, is_admin)
        for subject, relation, obj in TUPLES:
            await pg.insert_tuple(conn, subject, relation, obj)


async def main(settings: Settings) -> None:
    async with await AsyncConnection.connect(
        settings.pg_dsn, connect_timeout=int(settings.store_timeout_s)
    ) as conn:
        await seed(conn)
    log("seeded", job_id="seed", api_keys=len(API_KEYS), tuples=len(TUPLES))


if __name__ == "__main__":
    asyncio.run(main(Settings()), loop_factory=asyncio.SelectorEventLoop)
