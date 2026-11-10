import asyncio
import socket
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.ids import doc_id
from rag.common.log import log
from rag.common.models import make_embedder
from rag.worker.chunk import chunk
from rag.worker.index import Deps, cleanup, commit, emit, hash_file, index_new, stamp
from rag.worker.parse import parse
from rag.worker.watch import reconcile, watch


class Job(BaseModel):
    job_id: str
    op: Literal["upsert", "delete"]
    path: str


async def upsert(deps: Deps, path: str, job_id: str) -> None:
    file = Path(deps.settings.corpus_dir) / path
    if not file.is_file():
        log("upsert_skipped", job_id=job_id, path=path, reason="file is gone")
        return
    data = file.read_bytes()
    doc = doc_id(path)
    content_hash = hash_file(data)
    async with deps.pg.connection() as conn:
        row = await pg.get_document(conn, doc)
    if row is not None and row[0] == content_hash and not row[2]:
        log("upsert_skipped", job_id=job_id, path=path, reason="unchanged")
        return
    s = deps.settings
    parents = chunk(parse(path, data), s.parent_tokens, s.child_tokens, deps.embedder.count_tokens)
    async with deps.pg.connection() as conn:
        principals = await stamp(conn, path, doc)
    version = row[1] + 1 if row else 1
    chunks = await index_new(deps, doc, version, parents, principals)
    await commit(deps, doc, path, content_hash, version, parents)
    await cleanup(deps, doc, version)
    await emit(deps, "doc_changed", doc)
    log("document_indexed", job_id=job_id, path=path, version=version, chunks=chunks)


async def delete(deps: Deps, path: str, job_id: str) -> None:
    doc = doc_id(path)
    async with deps.pg.connection() as conn, conn.transaction():
        if await pg.get_document(conn, doc) is None:
            log("delete_skipped", job_id=job_id, path=path, reason="never ingested")
            return
        await pg.mark_document_deleted(conn, doc)
        await pg.delete_parents(conn, doc)
        await pg.delete_subject_tuples(conn, f"doc:{doc}", "parent")
    await qdrant_.delete_chunks(deps.qdrant, deps.settings, str(doc))
    await opensearch_.delete_chunks(deps.opensearch, deps.settings, str(doc))
    await emit(deps, "doc_deleted", doc)
    log("document_deleted", job_id=job_id, path=path)


async def handle(deps: Deps, message: redis_.Message) -> None:
    """Run one job; acknowledge it on success, dead-letter it after its last allowed attempt."""
    message_id, fields = message
    try:
        job = Job.model_validate(fields)
        async with deps.lock:
            await (upsert if job.op == "upsert" else delete)(deps, job.path, job.job_id)
    except Exception as exc:
        delivered = await redis_.deliveries(deps.redis, message_id)
        log("job_failed", job_id=fields.get("job_id"), error=repr(exc), deliveries=delivered)
        if delivered >= deps.settings.max_deliveries:
            await redis_.dead_letter(deps.redis, message_id, fields, repr(exc), delivered)
            log("job_dead_lettered", job_id=fields.get("job_id"), deliveries=delivered)
        return
    await redis_.ack_job(deps.redis, message_id)


async def consume_once(deps: Deps, consumer: str) -> int:
    """Handle the new jobs that arrive within one blocking read; returns how many."""
    messages = await redis_.read_jobs(deps.redis, consumer, deps.settings.worker_block_ms)
    for message in messages:
        await handle(deps, message)
    return len(messages)


async def reclaim_once(deps: Deps, consumer: str) -> int:
    """Retry jobs left unacknowledged for `RECLAIM_IDLE_MS`; returns how many."""
    messages = await redis_.claim_idle_jobs(deps.redis, consumer, deps.settings.reclaim_idle_ms)
    for message in messages:
        await handle(deps, message)
    return len(messages)


async def consume_loop(deps: Deps, consumer: str) -> None:
    while True:
        await consume_once(deps, consumer)


async def reclaim_loop(deps: Deps, consumer: str) -> None:
    while True:
        await asyncio.sleep(deps.settings.reclaim_idle_ms / 1000)
        await reclaim_once(deps, consumer)


async def open_deps(settings: Settings) -> Deps:
    return Deps(
        settings=settings,
        pg=await pg.open_pool(settings),
        redis=redis_.connect(settings),
        qdrant=qdrant_.connect(settings),
        opensearch=opensearch_.connect(settings),
        embedder=make_embedder(settings),
    )


async def close_deps(deps: Deps) -> None:
    await deps.pg.close()
    await deps.redis.aclose()
    await deps.qdrant.close()
    await deps.opensearch.close()


async def run(settings: Settings) -> None:
    """Reconcile the corpus, then watch it and process jobs until interrupted."""
    deps = await open_deps(settings)
    consumer = socket.gethostname()
    try:
        await redis_.ensure_jobs_group(deps.redis)
        await reconcile(settings, deps.pg, deps.redis)
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(watch(settings, deps.redis))
            tasks.create_task(consume_loop(deps, consumer))
            tasks.create_task(reclaim_loop(deps, consumer))
    finally:
        await close_deps(deps)


if __name__ == "__main__":
    asyncio.run(run(Settings()), loop_factory=asyncio.SelectorEventLoop)
