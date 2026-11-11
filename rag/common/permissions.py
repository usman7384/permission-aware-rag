from pathlib import PurePosixPath
from uuid import UUID

from psycopg import AsyncConnection

from rag.common import pg
from rag.common.keys import CHANGES_STREAM, tuple_event

ROOT_FOLDER = "_root"


def folder_of(path: str) -> str:
    """Folder of a corpus-relative path; files directly under CORPUS_DIR are in `_root`."""
    parent = PurePosixPath(path).parent.as_posix()
    return ROOT_FOLDER if parent == "." else parent


def parent_tuples(path: str, doc: UUID) -> list[tuple[str, str, str]]:
    """The document's `parent` tuple and one per nested folder above it."""
    folder = PurePosixPath(folder_of(path))
    found = [(f"doc:{doc}", "parent", f"folder:{folder}")]
    while folder.parent != PurePosixPath("."):
        found.append((f"folder:{folder}", "parent", f"folder:{folder.parent}"))
        folder = folder.parent
    return found


async def expand_principals(conn: AsyncConnection, user: str) -> list[str]:
    return await pg.select_principals(conn, user)


async def document_readers(conn: AsyncConnection, doc: UUID) -> list[str]:
    return await pg.select_document_readers(conn, f"doc:{doc}")


async def readable_docs(
    conn: AsyncConnection, principals: list[str], doc_ids: list[str]
) -> set[str]:
    """The subset of `doc_ids` that `principals` may read, from the live tuples."""
    if not doc_ids:
        return set()
    return await pg.select_readable_docs(conn, principals, doc_ids)


async def change_tuple(
    conn: AsyncConnection, op: str, subject: str, relation: str, obj: str
) -> None:
    """Add or remove the tuple and record its change event in the same transaction."""
    async with conn.transaction():
        write = pg.insert_tuple if op == "add" else pg.delete_tuple
        await write(conn, subject, relation, obj)
        await pg.insert_outbox(conn, CHANGES_STREAM, tuple_event(subject, relation, obj))
