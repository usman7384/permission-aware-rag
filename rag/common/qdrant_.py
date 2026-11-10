from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from rag.common.config import Settings


def connect(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=int(settings.store_timeout_s))


async def ping(client: AsyncQdrantClient) -> None:
    await client.get_collections()


async def ensure_collections(client: AsyncQdrantClient, settings: Settings) -> None:
    """Create the chunks and answer-cache collections and their payload indexes if missing."""
    indexed_fields = {
        settings.chunks_name: ["principals", "doc_id"],
        settings.answer_cache_name: ["fingerprint", "doc_ids"],
    }
    for name, fields in indexed_fields.items():
        if not await client.collection_exists(name):
            await client.create_collection(
                name, vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE)
            )
        for field in fields:
            await client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)


async def upsert_chunks(
    client: AsyncQdrantClient,
    settings: Settings,
    payloads: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    points = [
        PointStruct(id=payload["chunk_id"], vector=vector, payload=payload)
        for payload, vector in zip(payloads, vectors, strict=True)
    ]
    await client.upsert(settings.chunks_name, points=points, wait=True)


def _chunk_filter(doc_id: str, lt: int | None, gte: int | None) -> Filter:
    conditions = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if lt is not None or gte is not None:
        conditions.append(FieldCondition(key="version", range=Range(lt=lt, gte=gte)))
    return Filter(must=conditions)


async def delete_chunks(
    client: AsyncQdrantClient,
    settings: Settings,
    doc_id: str,
    lt: int | None = None,
    gte: int | None = None,
) -> None:
    """Delete a document's chunks, optionally only those with version < `lt` or >= `gte`."""
    await client.delete(
        settings.chunks_name, points_selector=_chunk_filter(doc_id, lt, gte), wait=True
    )


async def set_principals(
    client: AsyncQdrantClient, settings: Settings, doc_id: str, principals: list[str]
) -> None:
    """Overwrite `principals` on every chunk of the document, whatever its version."""
    await client.set_payload(
        settings.chunks_name,
        payload={"principals": principals},
        points=_chunk_filter(doc_id, None, None),
        wait=True,
    )


async def search_chunks(
    client: AsyncQdrantClient,
    settings: Settings,
    vector: list[float],
    principals: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Payloads of the nearest chunks whose `principals` share at least one of `principals`."""
    allowed = Filter(must=[FieldCondition(key="principals", match=MatchAny(any=principals))])
    response = await client.query_points(
        settings.chunks_name, query=vector, query_filter=allowed, limit=limit, with_payload=True
    )
    return [point.payload for point in response.points]
