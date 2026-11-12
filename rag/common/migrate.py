import asyncio
from pathlib import Path

from psycopg import AsyncConnection

from rag.common import opensearch_, pg, qdrant_
from rag.common.config import Settings
from rag.common.log import log

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def migrate(settings: Settings, job_id: str) -> None:
    async with await AsyncConnection.connect(
        settings.pg_dsn, connect_timeout=int(settings.store_timeout_s)
    ) as conn:
        applied = await pg.apply_migrations(conn, MIGRATIONS_DIR)
    log("migrations_applied", job_id=job_id, files=applied)
    qdrant = qdrant_.connect(settings)
    try:
        await qdrant_.ensure_collections(qdrant, settings)
    finally:
        await qdrant.close()
    opensearch = opensearch_.connect(settings)
    try:
        await opensearch_.ensure_index(opensearch, settings)
    finally:
        await opensearch.close()
    log("indexes_ready", job_id=job_id, chunks=settings.chunks_name)


if __name__ == "__main__":
    asyncio.run(migrate(Settings(), job_id="migrate"), loop_factory=asyncio.SelectorEventLoop)
