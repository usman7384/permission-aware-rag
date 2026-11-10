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
