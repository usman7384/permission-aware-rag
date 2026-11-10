from pathlib import PurePosixPath
from uuid import UUID

from psycopg import AsyncConnection

from rag.common import pg

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


async def document_readers(conn: AsyncConnection, doc: UUID) -> list[str]:
    return await pg.select_document_readers(conn, f"doc:{doc}")
