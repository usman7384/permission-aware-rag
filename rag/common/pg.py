from pathlib import Path
from uuid import UUID

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


async def get_document(conn: AsyncConnection, doc_id: UUID) -> tuple[str, int, bool] | None:
    """(content_hash, version, is_deleted) of a document, or None if it was never ingested."""
    cur = await conn.execute(
        "SELECT content_hash, version, deleted_at IS NOT NULL FROM documents WHERE doc_id = %s",
        (doc_id,),
    )
    return await cur.fetchone()


async def upsert_document(
    conn: AsyncConnection, doc_id: UUID, path: str, folder: str, content_hash: str, version: int
) -> None:
    await conn.execute(
        "INSERT INTO documents (doc_id, path, folder, content_hash, version) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (doc_id) DO UPDATE SET "
        "folder = EXCLUDED.folder, content_hash = EXCLUDED.content_hash, "
        "version = EXCLUDED.version, updated_at = now(), deleted_at = NULL",
        (doc_id, path, folder, content_hash, version),
    )


async def insert_parents(
    conn: AsyncConnection, rows: list[tuple[UUID, UUID, int, int, str | None, str]]
) -> None:
    """Rows are (parent_id, doc_id, version, ordinal, heading, text)."""
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO parents (parent_id, doc_id, version, ordinal, heading, text) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (parent_id) DO NOTHING",
            rows,
        )


async def delete_parents(
    conn: AsyncConnection, doc_id: UUID, below_version: int | None = None
) -> None:
    """Delete a document's parents, or only those older than `below_version`."""
    await conn.execute(
        "DELETE FROM parents WHERE doc_id = %s AND (%s::int IS NULL OR version < %s::int)",
        (doc_id, below_version, below_version),
    )


async def mark_document_deleted(conn: AsyncConnection, doc_id: UUID) -> None:
    await conn.execute(
        "UPDATE documents SET deleted_at = now() WHERE doc_id = %s AND deleted_at IS NULL",
        (doc_id,),
    )


async def delete_subject_tuples(conn: AsyncConnection, subject: str, relation: str) -> None:
    await conn.execute(
        "DELETE FROM tuples WHERE subject = %s AND relation = %s", (subject, relation)
    )


async def live_document_paths(conn: AsyncConnection) -> list[str]:
    cur = await conn.execute("SELECT path FROM documents WHERE deleted_at IS NULL ORDER BY path")
    return [path for (path,) in await cur.fetchall()]


async def select_document_readers(conn: AsyncConnection, obj: str) -> list[str]:
    """Subjects with `viewer` on `obj` or on any folder above it through `parent` tuples."""
    cur = await conn.execute(
        "WITH RECURSIVE anc(obj) AS ("
        "  SELECT %(doc)s::text"
        "  UNION"
        "  SELECT t.object FROM tuples t JOIN anc ON t.subject = anc.obj"
        "  WHERE t.relation = 'parent'"
        ") "
        "SELECT DISTINCT t.subject FROM tuples t JOIN anc ON t.object = anc.obj "
        "WHERE t.relation = 'viewer' ORDER BY t.subject",
        {"doc": obj},
    )
    return [subject for (subject,) in await cur.fetchall()]
