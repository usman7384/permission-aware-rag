from pathlib import Path

from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis
from watchfiles import Change, awatch

from rag.common import pg, redis_
from rag.common.config import Settings
from rag.common.log import log

WATCHED_SUFFIXES = (".md", ".txt")


def corpus_files(root: Path) -> list[str]:
    """Watched files under `root`, as sorted relative paths with `/` separators."""
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in WATCHED_SUFFIXES
    )


async def _enqueue(redis: Redis, op: str, path: str) -> None:
    job_id = await redis_.enqueue_job(redis, op, path)
    log("job_enqueued", job_id=job_id, op=op, path=path)


async def reconcile(settings: Settings, pool: AsyncConnectionPool, redis: Redis) -> None:
    """Enqueue an upsert for every file and a delete for every live document whose file is gone."""
    files = corpus_files(Path(settings.corpus_dir).resolve())
    async with pool.connection() as conn:
        live = await pg.live_document_paths(conn)
    for path in files:
        await _enqueue(redis, "upsert", path)
    for path in sorted(set(live) - set(files)):
        await _enqueue(redis, "delete", path)


async def watch(settings: Settings, redis: Redis) -> None:
    root = Path(settings.corpus_dir).resolve()
    async for changes in awatch(root):
        for change, raw in sorted(changes):
            path = Path(raw).resolve()
            if path.suffix in WATCHED_SUFFIXES:
                op = "delete" if change == Change.deleted else "upsert"
                await _enqueue(redis, op, path.relative_to(root).as_posix())
