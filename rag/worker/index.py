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

from rag.common import opensearch_, pg, qdrant_
from rag.common.config import Settings
from rag.common.ids import chunk_id, parent_id
from rag.common.keys import CHANGES_STREAM, doc_event
from rag.common.models import LLM, Embedder
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
    llm: LLM
    # Jobs run one at a time so two versions of a document are never written at once.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def hash_file(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def stamp(conn: AsyncConnection, path: str, doc: UUID) -> list[str]:
    for subject, relation, obj in parent_tuples(path, doc):
        await pg.insert_tuple(conn, subject, relation, obj)
    return await document_readers(conn, doc)


def chunk_payloads(
    doc: UUID,
    version: int,
    parents: list[Parent],
    contexts: list[str | None],
    principals: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads, texts = [], []
    for p, (parent, context) in enumerate(zip(parents, contexts, strict=True)):
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
            texts.append(child if context is None else f"{context}\n{child}")
    return payloads, texts


async def index_new(
    deps: Deps,
    doc: UUID,
    version: int,
    parents: list[Parent],
    contexts: list[str | None],
    principals: list[str],
) -> int:
    s = deps.settings
    payloads, texts = chunk_payloads(doc, version, parents, contexts, principals)
    # Drop chunks left by an earlier failed attempt at this version.
    await qdrant_.delete_chunks(deps.qdrant, s, str(doc), gte=version)
    await opensearch_.delete_chunks(deps.opensearch, s, str(doc), gte=version)
    if payloads:
        vectors = await asyncio.to_thread(deps.embedder.embed_documents, texts)
        await qdrant_.upsert_chunks(deps.qdrant, s, payloads, vectors)
        await opensearch_.index_chunks(deps.opensearch, s, payloads, texts)
    return len(payloads)


async def commit(
    deps: Deps,
    doc: UUID,
    path: str,
    content_hash: str,
    version: int,
    parents: list[Parent],
    contexts: list[str | None],
) -> None:
    rows = [
        (parent_id(doc, version, p), doc, version, p, parent.heading, context, parent.text)
        for p, (parent, context) in enumerate(zip(parents, contexts, strict=True))
    ]
    async with deps.pg.connection() as conn, conn.transaction():
        await pg.upsert_document(conn, doc, path, folder_of(path), content_hash, version)
        await pg.insert_parents(conn, rows)
        await pg.insert_outbox(conn, CHANGES_STREAM, doc_event("doc_changed", [str(doc)]))


async def cleanup(deps: Deps, doc: UUID, version: int) -> None:
    await qdrant_.delete_chunks(deps.qdrant, deps.settings, str(doc), lt=version)
    await opensearch_.delete_chunks(deps.opensearch, deps.settings, str(doc), lt=version)
    async with deps.pg.connection() as conn:
        await pg.delete_parents(conn, doc, below_version=version)
