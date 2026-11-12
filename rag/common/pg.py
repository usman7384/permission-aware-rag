from pathlib import Path
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection, sql
from psycopg.types.json import Jsonb
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


async def delete_tuple(conn: AsyncConnection, subject: str, relation: str, obj: str) -> None:
    await conn.execute(
        "DELETE FROM tuples WHERE subject = %s AND relation = %s AND object = %s",
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
    conn: AsyncConnection, rows: list[tuple[UUID, UUID, int, int, str | None, str | None, str]]
) -> None:
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO parents (parent_id, doc_id, version, ordinal, heading, context, text) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (parent_id) DO NOTHING",
            rows,
        )


async def delete_parents(
    conn: AsyncConnection, doc_id: UUID, below_version: int | None = None
) -> None:
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


async def select_live_documents_beneath(conn: AsyncConnection, obj: str) -> list[UUID]:
    cur = await conn.execute(
        "WITH RECURSIVE below(obj) AS ("
        "  SELECT %(obj)s::text"
        "  UNION"
        "  SELECT t.subject FROM tuples t JOIN below ON t.object = below.obj"
        "  WHERE t.relation = 'parent'"
        ") "
        "SELECT d.doc_id FROM documents d JOIN below ON below.obj = 'doc:' || d.doc_id "
        "WHERE d.deleted_at IS NULL ORDER BY d.doc_id",
        {"obj": obj},
    )
    return [doc for (doc,) in await cur.fetchall()]


async def select_principals(conn: AsyncConnection, user: str) -> list[str]:
    cur = await conn.execute(
        "WITH RECURSIVE g(principal) AS ("
        "  SELECT %(user)s::text"
        "  UNION"
        "  SELECT t.object FROM tuples t JOIN g ON t.subject = g.principal"
        "  WHERE t.relation = 'member'"
        ") "
        "SELECT principal FROM g ORDER BY principal",
        {"user": user},
    )
    return [principal for (principal,) in await cur.fetchall()]


async def select_readable_docs(
    conn: AsyncConnection, principals: list[str], doc_ids: list[str]
) -> set[str]:
    cur = await conn.execute(
        "WITH RECURSIVE anc(doc_id, obj) AS ("
        "  SELECT d, 'doc:' || d FROM unnest(%(docs)s::uuid[]) AS d"
        "  UNION"
        "  SELECT anc.doc_id, t.object FROM tuples t JOIN anc ON t.subject = anc.obj"
        "  WHERE t.relation = 'parent'"
        ") "
        "SELECT DISTINCT anc.doc_id::text FROM anc JOIN tuples t ON t.object = anc.obj "
        "WHERE t.relation = 'viewer' AND t.subject = ANY(%(principals)s::text[])",
        {"docs": doc_ids, "principals": principals},
    )
    return {doc for (doc,) in await cur.fetchall()}


async def live_document_versions(conn: AsyncConnection, doc_ids: list[str]) -> dict[str, int]:
    cur = await conn.execute(
        "SELECT doc_id::text, version FROM documents "
        "WHERE doc_id = ANY(%s::uuid[]) AND deleted_at IS NULL",
        (doc_ids,),
    )
    return dict(await cur.fetchall())


async def insert_outbox(conn: AsyncConnection, stream: str, payload: dict[str, str]) -> None:
    await conn.execute(
        "INSERT INTO outbox (stream, payload) VALUES (%s, %s)", (stream, Jsonb(payload))
    )


async def claim_outbox(conn: AsyncConnection, limit: int) -> list[tuple[int, str, dict[str, str]]]:
    cur = await conn.execute(
        "SELECT id, stream, payload FROM outbox WHERE published_at IS NULL "
        "ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED",
        (limit,),
    )
    return await cur.fetchall()


async def mark_outbox_published(conn: AsyncConnection, ids: list[int]) -> None:
    await conn.execute("UPDATE outbox SET published_at = now() WHERE id = ANY(%s)", (ids,))


async def get_api_key(conn: AsyncConnection, key_hash: str) -> tuple[str, bool] | None:
    cur = await conn.execute(
        "SELECT user_id, is_admin FROM api_keys WHERE key_hash = %s", (key_hash,)
    )
    return await cur.fetchone()


async def get_parents(
    conn: AsyncConnection, parent_ids: list[str]
) -> list[tuple[str, str, int, str, str | None, str]]:
    cur = await conn.execute(
        "SELECT p.parent_id::text, p.doc_id::text, p.version, d.path, p.heading, p.text "
        "FROM parents p JOIN documents d ON d.doc_id = p.doc_id "
        "WHERE p.parent_id = ANY(%s::uuid[])",
        (parent_ids,),
    )
    return await cur.fetchall()


async def insert_turn(
    conn: AsyncConnection, session_id: str, user_id: str, role: str, text: str
) -> None:
    await conn.execute(
        "INSERT INTO turns (session_id, user_id, role, text) VALUES (%s, %s, %s, %s)",
        (session_id, user_id, role, text),
    )


async def recent_turns(
    conn: AsyncConnection, session_id: str, user_id: str, limit: int
) -> list[tuple[str, str]]:
    cur = await conn.execute(
        "SELECT role, text FROM ("
        "  SELECT turn_id, role, text FROM turns WHERE session_id = %s AND user_id = %s"
        "  ORDER BY turn_id DESC LIMIT %s"
        ") recent ORDER BY turn_id",
        (session_id, user_id, limit),
    )
    return await cur.fetchall()


async def insert_request(
    conn: AsyncConnection,
    request_id: str,
    user_id: str,
    session_id: str,
    question: str,
    rewritten: str | None,
    cache_tier: str,
    abstained: bool,
    degraded: list[str],
    doc_ids: list[str],
    timings: dict[str, float],
    answer: str,
) -> None:
    await conn.execute(
        "INSERT INTO requests (request_id, user_id, session_id, question, rewritten, "
        "cache_tier, abstained, degraded, doc_ids, timings, answer) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s) "
        "ON CONFLICT (request_id) DO NOTHING",
        (
            request_id,
            user_id,
            session_id,
            question,
            rewritten,
            cache_tier,
            abstained,
            degraded,
            doc_ids,
            Jsonb(timings),
            answer,
        ),
    )


async def set_request_timings(
    conn: AsyncConnection, request_id: str, timings: dict[str, float]
) -> None:
    await conn.execute(
        "UPDATE requests SET timings = %s WHERE request_id = %s", (Jsonb(timings), request_id)
    )


async def insert_feedback(
    conn: AsyncConnection, request_id: str, user_id: str, rating: int, comment: str | None
) -> bool:
    cur = await conn.execute(
        "INSERT INTO feedback (request_id, rating, comment) "
        "SELECT request_id, %s, %s FROM requests WHERE request_id = %s AND user_id = %s",
        (rating, comment, request_id, user_id),
    )
    return cur.rowcount == 1


async def insert_eval_run(
    conn: AsyncConnection, run_id: str, name: str, config: dict[str, Any], metrics: dict[str, Any]
) -> None:
    await conn.execute(
        "INSERT INTO eval_runs (run_id, name, config, metrics) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (run_id) DO NOTHING",
        (run_id, name, Jsonb(config), Jsonb(metrics)),
    )


async def ensure_database(dsn: str, name: str, timeout_s: float) -> None:
    async with await AsyncConnection.connect(
        dsn, autocommit=True, connect_timeout=int(timeout_s)
    ) as conn:
        cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if await cur.fetchone() is None:
            await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
