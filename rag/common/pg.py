from pathlib import Path

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from rag.common.config import Settings


async def open_pool(settings: Settings) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        settings.pg_dsn,
        open=False,
        timeout=settings.store_timeout_s,
        kwargs={"connect_timeout": int(settings.store_timeout_s)},
    )
    await pool.open(wait=False)
    return pool


async def ping(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute("SELECT 1")


async def apply_migrations(conn: AsyncConnection, directory: Path) -> list[str]:
    """Run every .sql file in name order; migrations are written to be re-runnable."""
    applied = []
    for path in sorted(directory.glob("*.sql")):
        async with conn.transaction():
            await conn.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
    return applied


async def insert_tuple(conn: AsyncConnection, subject: str, relation: str, obj: str) -> None:
    await conn.execute(
        "INSERT INTO tuples (subject, relation, object) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (subject, relation, obj),
    )


async def upsert_api_key(
    conn: AsyncConnection, key_hash: str, user_id: str, is_admin: bool
) -> None:
    await conn.execute(
        "INSERT INTO api_keys (key_hash, user_id, is_admin) VALUES (%s, %s, %s) "
        "ON CONFLICT (key_hash) DO UPDATE SET user_id = EXCLUDED.user_id, "
        "is_admin = EXCLUDED.is_admin",
        (key_hash, user_id, is_admin),
    )
