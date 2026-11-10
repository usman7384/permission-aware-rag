from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

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
