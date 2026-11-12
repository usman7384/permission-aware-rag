import asyncio
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.faults import FaultyLLM
from rag.common.ids import doc_id
from rag.common.keys import CHANGES_GROUP, CHANGES_STREAM, JOBS_GROUP, JOBS_STREAM, doc_event
from rag.common.log import log
from rag.common.models import make_embedder, make_llm
from rag.worker import invalidate, relay
from rag.worker.chunk import chunk
from rag.worker.contextualize import contextualize
from rag.worker.index import Deps, cleanup, commit, hash_file, index_new, stamp
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
    s, count = deps.settings, deps.embedder.count_tokens
    sections = await asyncio.to_thread(parse, path, data)
    parents = chunk(sections, s.parent_tokens, s.child_tokens, count)
    contexts: list[str | None] = [None] * len(parents)
    if s.contextual_chunking:
        document = "".join(section.text for section in sections)
        contexts = list(await contextualize(deps.llm, s, document, parents, count))
    async with deps.pg.connection() as conn:
        principals = await stamp(conn, path, doc)
    version = row[1] + 1 if row else 1
    chunks = await index_new(deps, doc, version, parents, contexts, principals)
    await commit(deps, doc, path, content_hash, version, parents, contexts)
    await cleanup(deps, doc, version)
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
        await pg.insert_outbox(conn, CHANGES_STREAM, doc_event("doc_deleted", [str(doc)]))
    await qdrant_.delete_chunks(deps.qdrant, deps.settings, str(doc))
    await opensearch_.delete_chunks(deps.opensearch, deps.settings, str(doc))
    log("document_deleted", job_id=job_id, path=path)


async def handle(deps: Deps, message: redis_.Message) -> None:
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
    await redis_.ack(deps.redis, JOBS_STREAM, JOBS_GROUP, message_id)


async def consume_once(deps: Deps, consumer: str) -> int:
    messages = await redis_.read_new(
        deps.redis, JOBS_STREAM, JOBS_GROUP, consumer, deps.settings.worker_block_ms
    )
    for message in messages:
        await handle(deps, message)
    return len(messages)


async def reclaim_once(deps: Deps, consumer: str) -> int:
    messages = await redis_.claim_idle(
        deps.redis, JOBS_STREAM, JOBS_GROUP, consumer, deps.settings.reclaim_idle_ms
    )
    for message in messages:
        await handle(deps, message)
    return len(messages)


Step = Callable[[Deps, str], Awaitable[int]]


async def consume_loop(deps: Deps, consumer: str, consume: Step) -> None:
    while True:
        await consume(deps, consumer)


async def reclaim_loop(deps: Deps, consumer: str, reclaim: Step) -> None:
    while True:
        await asyncio.sleep(deps.settings.reclaim_idle_ms / 1000)
        await reclaim(deps, consumer)


async def open_deps(settings: Settings) -> Deps:
    redis = redis_.connect(settings)
    return Deps(
        settings=settings,
        pg=await pg.open_pool(settings),
        redis=redis,
        qdrant=qdrant_.connect(settings),
        opensearch=opensearch_.connect(settings),
        embedder=make_embedder(settings),
        llm=FaultyLLM(make_llm(settings), redis, settings.fault_slow_ms),
    )


async def close_deps(deps: Deps) -> None:
    await deps.pg.close()
    await deps.redis.aclose()
    await deps.qdrant.close()
    await deps.opensearch.close()


async def run(settings: Settings) -> None:
    deps = await open_deps(settings)
    consumer = socket.gethostname()
    try:
        await redis_.ensure_group(deps.redis, JOBS_STREAM, JOBS_GROUP)
        await redis_.ensure_group(deps.redis, CHANGES_STREAM, CHANGES_GROUP)
        await reconcile(settings, deps.pg, deps.redis)
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(watch(settings, deps.redis))
            tasks.create_task(consume_loop(deps, consumer, consume_once))
            tasks.create_task(reclaim_loop(deps, consumer, reclaim_once))
            tasks.create_task(consume_loop(deps, consumer, invalidate.consume_once))
            tasks.create_task(reclaim_loop(deps, consumer, invalidate.reclaim_once))
            tasks.create_task(relay.relay_loop(deps))
    finally:
        await close_deps(deps)


if __name__ == "__main__":
    asyncio.run(run(Settings()), loop_factory=asyncio.SelectorEventLoop)
