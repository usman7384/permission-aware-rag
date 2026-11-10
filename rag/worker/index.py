import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.ids import chunk_id, parent_id
from rag.common.models import Embedder
from rag.common.permissions import document_readers, folder_of, parent_tuples
from rag.worker.chunk import Parent


@dataclass
class Deps:
    settings: Settings
    pg: AsyncConnectionPool
    redis: Redis
    qdrant: AsyncQdrantClient
    opensearch: AsyncOpenSearch
    embedder: Embedder
    # Jobs run one at a time so two versions of a document are never written at once.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def hash_file(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def stamp(conn: AsyncConnection, path: str, doc: UUID) -> list[str]:
    """Ensure the document's `parent` tuples exist, then return its readers."""
    for subject, relation, obj in parent_tuples(path, doc):
        await pg.insert_tuple(conn, subject, relation, obj)
    return await document_readers(conn, doc)


def _chunk_payloads(
    doc: UUID, version: int, parents: list[Parent], principals: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads, texts = [], []
    for p, parent in enumerate(parents):
        for c, child in enumerate(parent.children):
            payloads.append(
                {
                    "chunk_id": str(chunk_id(doc, version, p, c)),
                    "parent_id": str(parent_id(doc, version, p)),
                    "doc_id": str(doc),
                    "version": version,
                    "principals": principals,
                }
            )
            texts.append(child)
    return payloads, texts


async def index_new(
    deps: Deps, doc: UUID, version: int, parents: list[Parent], principals: list[str]
) -> int:
    """Embed the children and write them to Qdrant and OpenSearch; returns the chunk count."""
    s = deps.settings
    payloads, texts = _chunk_payloads(doc, version, parents, principals)
    # Chunks left by an earlier failed attempt at this version must not survive the commit.
    await qdrant_.delete_chunks(deps.qdrant, s, str(doc), gte=version)
    await opensearch_.delete_chunks(deps.opensearch, s, str(doc), gte=version)
    if payloads:
        vectors = await asyncio.to_thread(deps.embedder.embed_documents, texts)
        await qdrant_.upsert_chunks(deps.qdrant, s, payloads, vectors)
        await opensearch_.index_chunks(deps.opensearch, s, payloads, texts)
    return len(payloads)


async def commit(
    deps: Deps, doc: UUID, path: str, content_hash: str, version: int, parents: list[Parent]
) -> None:
    rows = [
        (parent_id(doc, version, p), doc, version, p, parent.heading, parent.text)
        for p, parent in enumerate(parents)
    ]
    async with deps.pg.connection() as conn, conn.transaction():
        await pg.upsert_document(conn, doc, path, folder_of(path), content_hash, version)
        await pg.insert_parents(conn, rows)


async def cleanup(deps: Deps, doc: UUID, version: int) -> None:
    """Delete the document's chunks and parents older than `version`."""
    await qdrant_.delete_chunks(deps.qdrant, deps.settings, str(doc), lt=version)
    await opensearch_.delete_chunks(deps.opensearch, deps.settings, str(doc), lt=version)
    async with deps.pg.connection() as conn:
        await pg.delete_parents(conn, doc, below_version=version)


async def emit(deps: Deps, event_type: str, doc: UUID) -> None:
    await redis_.emit_change(deps.redis, event_type, [str(doc)])
