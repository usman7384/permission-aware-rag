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


async def set_principals(
    client: AsyncOpenSearch, settings: Settings, doc_id: str, principals: list[str]
) -> None:
    """Overwrite `principals` on every chunk of the document, whatever its version."""
    body = {
        "query": {"term": {"doc_id": doc_id}},
        "script": {
            "source": "ctx._source.principals = params.principals",
            "params": {"principals": principals},
        },
    }
    response = await client.update_by_query(index=settings.chunks_name, body=body, refresh=True)
    if response["failures"]:
        raise RuntimeError(f"OpenSearch principals update failed: {response['failures']}")


async def search_chunks(
    client: AsyncOpenSearch, settings: Settings, text: str, principals: list[str], size: int
) -> list[dict[str, Any]]:
    """BM25 matches on `text` among chunks whose `principals` share one of `principals`."""
    body = {
        "size": size,
        "_source": ["chunk_id", "parent_id", "doc_id", "version"],
        "query": {
            "bool": {
                "must": [{"match": {"text": text}}],
                "filter": [{"terms": {"principals": principals}}],
            }
        },
    }
    response = await client.search(index=settings.chunks_name, body=body)
    return [hit["_source"] for hit in response["hits"]["hits"]]


async def get_texts(
    client: AsyncOpenSearch, settings: Settings, chunk_ids: list[str]
) -> dict[str, str]:
    """Indexed text by chunk_id; chunks no longer in the index are absent."""
    response = await client.mget(
        index=settings.chunks_name, body={"ids": chunk_ids}, _source=["text"]
    )
    return {doc["_id"]: doc["_source"]["text"] for doc in response["docs"] if doc.get("found")}


async def scroll_chunks(client: AsyncOpenSearch, settings: Settings) -> list[dict[str, Any]]:
    """Every indexed chunk: its payload fields and its `text`."""
    response = await client.search(
        index=settings.chunks_name, body={"query": {"match_all": {}}}, scroll="2m", size=1000
    )
    chunks, scroll_id = [], response["_scroll_id"]
    try:
        while hits := response["hits"]["hits"]:
            chunks += [hit["_source"] for hit in hits]
            response = await client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
    finally:
        await client.clear_scroll(scroll_id=scroll_id)
    return chunks
