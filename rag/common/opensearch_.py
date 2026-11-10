from typing import Any

from opensearchpy import AsyncOpenSearch

from rag.common.config import Settings

CHUNKS_MAPPING = {
    "properties": {
        "text": {"type": "text", "analyzer": "standard"},
        "chunk_id": {"type": "keyword"},
        "parent_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "version": {"type": "integer"},
        "principals": {"type": "keyword"},
    }
}


def connect(settings: Settings) -> AsyncOpenSearch:
    return AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=settings.store_timeout_s)


async def ping(client: AsyncOpenSearch) -> None:
    await client.cluster.health()


async def ensure_index(client: AsyncOpenSearch, settings: Settings) -> None:
    if not await client.indices.exists(index=settings.chunks_name):
        await client.indices.create(index=settings.chunks_name, body={"mappings": CHUNKS_MAPPING})


async def index_chunks(
    client: AsyncOpenSearch, settings: Settings, payloads: list[dict[str, Any]], texts: list[str]
) -> None:
    body: list[dict[str, Any]] = []
    for payload, text in zip(payloads, texts, strict=True):
        body.append({"index": {"_index": settings.chunks_name, "_id": payload["chunk_id"]}})
        body.append({**payload, "text": text})
    response = await client.bulk(body=body, refresh="true")
    if response["errors"]:
        raise RuntimeError(f"OpenSearch bulk index failed: {response['items']}")


async def delete_chunks(
    client: AsyncOpenSearch,
    settings: Settings,
    doc_id: str,
    lt: int | None = None,
    gte: int | None = None,
) -> None:
    """Delete a document's chunks, optionally only those with version < `lt` or >= `gte`."""
    conditions: list[dict[str, Any]] = [{"term": {"doc_id": doc_id}}]
    if lt is not None or gte is not None:
        bounds = {k: v for k, v in (("lt", lt), ("gte", gte)) if v is not None}
        conditions.append({"range": {"version": bounds}})
    await client.delete_by_query(
        index=settings.chunks_name,
        body={"query": {"bool": {"filter": conditions}}},
        refresh=True,
        conflicts="proceed",
    )
