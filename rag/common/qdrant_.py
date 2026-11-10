from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Memory,
    PayloadSchemaType,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from rag.common.config import Settings


def connect(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=int(settings.store_timeout_s))


async def ping(client: AsyncQdrantClient) -> None:
    await client.get_collections()


def chunk_vector_params(settings: Settings) -> VectorParams:
    """Chunk vectors, optionally quantized (kept in RAM) with the originals left on disk."""
    quantization: ScalarQuantization | BinaryQuantization | None = None
    if settings.qdrant_quantization == "int8":
        config = ScalarQuantizationConfig(type=ScalarType.INT8, memory=Memory.PINNED)
        quantization = ScalarQuantization(scalar=config)
    elif settings.qdrant_quantization == "binary":
        quantization = BinaryQuantization(binary=BinaryQuantizationConfig(memory=Memory.PINNED))
    return VectorParams(
        size=settings.embed_dim,
        distance=Distance.COSINE,
        quantization_config=quantization,
        memory=Memory.COLD if settings.qdrant_vectors_on_disk else None,
    )


async def ensure_collections(client: AsyncQdrantClient, settings: Settings) -> None:
    """Create the chunks and answer-cache collections and their payload indexes if missing."""
    collections = {
        settings.chunks_name: (chunk_vector_params(settings), ["principals", "doc_id"]),
        settings.answer_cache_name: (
            VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
            ["fingerprint", "doc_ids"],
        ),
    }
    for name, (vectors, fields) in collections.items():
        if not await client.collection_exists(name):
            await client.create_collection(name, vectors_config=vectors)
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


async def nearest_answer(
    client: AsyncQdrantClient, settings: Settings, vector: list[float], fingerprint: str
) -> tuple[str, float, dict[str, Any]] | None:
    """(point ID, similarity, payload) of the nearest cached answer with this fingerprint."""
    same_user = Filter(
        must=[FieldCondition(key="fingerprint", match=MatchValue(value=fingerprint))]
    )
    response = await client.query_points(
        settings.answer_cache_name, query=vector, query_filter=same_user, limit=1, with_payload=True
    )
    if not response.points:
        return None
    point = response.points[0]
    return str(point.id), point.score, point.payload or {}


async def upsert_answer(
    client: AsyncQdrantClient,
    settings: Settings,
    point_id: str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    point = PointStruct(id=point_id, vector=vector, payload=payload)
    await client.upsert(settings.answer_cache_name, points=[point], wait=True)


async def delete_answer(client: AsyncQdrantClient, settings: Settings, point_id: str) -> None:
    await client.delete(settings.answer_cache_name, points_selector=[point_id], wait=True)


async def delete_answers_of(
    client: AsyncQdrantClient, settings: Settings, doc_ids: list[str]
) -> None:
    """Delete every cached answer whose context used one of the documents."""
    used = Filter(must=[FieldCondition(key="doc_ids", match=MatchAny(any=doc_ids))])
    await client.delete(settings.answer_cache_name, points_selector=used, wait=True)
