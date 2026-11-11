import argparse
import asyncio
import contextlib
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from psycopg import AsyncConnection

from rag.api import pipeline
from rag.api.auth import Caller
from rag.api.retrieve import recheck, sparse
from rag.common import pg, redis_
from rag.common.config import Settings
from rag.common.ids import doc_id
from rag.common.keys import cache_key
from rag.evals.benchmark import BenchDoc, document_bytes, read_corpus
from rag.evals.experiments import common
from rag.evals.score import percentile
from rag.worker import main as worker

FOLDER = "fresh"


async def seconds_until(
    check: Callable[[], Awaitable[bool]], start: float, timeout_s: float, poll_s: float
) -> float:
    """Seconds from `start` (a `perf_counter` time) until `check` first returns true."""
    async with asyncio.timeout(timeout_s):
        while not await check():
            await asyncio.sleep(poll_s)
    return round(time.perf_counter() - start, 3)


def scratch_redis(url: str, database: int) -> str:
    """The Redis URL with another database number; refuses the URL's own database."""
    if database == int(urlsplit(url).path.lstrip("/") or 0):
        raise ValueError(f"--redis-db {database} is the configured database; pick a scratch one")
    return urlunsplit(urlsplit(url)._replace(path=f"/{database}"))


def path_of(doc: BenchDoc) -> str:
    return f"{FOLDER}/{doc.id}.txt"


async def cached_answer_key(deps: pipeline.Deps, user: str, question: str) -> str:
    """Ask the question through the full query path; returns the key of its cached answer."""
    async with deps.pg.connection() as conn:
        principals, fp = await pipeline.expand(conn, user)
    caller = Caller(user_id=user, is_admin=False, key_hash="")
    session, request_id = str(uuid.uuid4()), str(uuid.uuid4())
    async for _ in pipeline.ask(deps, caller, principals, fp, session, question, request_id, {}):
        pass
    key = cache_key(question, fp, deps.settings.index_version)
    if await redis_.get_answer(deps.redis, key) is None:
        raise RuntimeError(f"the answer to {question!r} was not cached (degraded or abstained)")
    return key


async def searchable(deps: pipeline.Deps, user: str, word: str, doc: str) -> bool:
    """Whether a keyword search for `word` returns the document's current version."""
    async with deps.pg.connection() as conn:
        principals, _ = await pipeline.expand(conn, user)
        hits = await sparse(deps.opensearch, deps.redis, deps.settings, word, principals)
        return any(c.doc_id == doc for c in await recheck(conn, hits, principals))


async def trial(deps: pipeline.Deps, args: argparse.Namespace, doc: BenchDoc) -> list[float]:
    """Cache an answer that used the document, append a new word to its file, then time how
    long until the word is searchable and the cached answer is evicted."""
    key = await cached_answer_key(deps, args.user, doc.title or doc.text[:80])
    word = f"fresh{uuid.uuid4().hex[:12]}"
    file = Path(deps.settings.corpus_dir) / path_of(doc)
    start = time.perf_counter()
    file.write_bytes(document_bytes(doc) + f"\n\nMarker {word}.".encode())

    async def evicted() -> bool:
        return await redis_.get_answer(deps.redis, key) is None

    found = seconds_until(
        lambda: searchable(deps, args.user, word, str(doc_id(path_of(doc)))),
        start,
        args.timeout_s,
        args.poll_ms / 1000,
    )
    gone = seconds_until(evicted, start, args.timeout_s, args.poll_ms / 1000)
    return list(await asyncio.gather(found, gone))


async def indexed(deps: pipeline.Deps, count: int) -> bool:
    async with deps.pg.connection() as conn:
        return len(await pg.live_document_paths(conn)) == count


async def run_trials(
    settings: Settings, args: argparse.Namespace, docs: list[BenchDoc]
) -> tuple[float, list[list[float]]]:
    """Empty the scratch Redis database, start the worker on the corpus folder, wait for its
    first index, then run the trials; returns the first index's seconds and each trial's."""
    deps = await pipeline.open_deps(settings)
    deps.llm = common.fake_llm_unless_configured(settings) or deps.llm
    await deps.redis.flushdb()
    running = asyncio.create_task(worker.run(settings))
    try:
        start = time.perf_counter()
        initial = await seconds_until(
            lambda: indexed(deps, len(docs)), start, args.timeout_s * 10, args.poll_ms / 1000
        )
        rows = [await trial(deps, args, docs[i % len(docs)]) for i in range(args.trials)]
    finally:
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
        await deps.redis.flushdb()
        await pipeline.close_deps(deps)
    return initial, rows


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    docs = read_corpus(args.benchmark, args.docs)
    corpus = Path(tempfile.mkdtemp(prefix="freshness-"))
    redis_url = scratch_redis(settings.redis_url, args.redis_db)
    s = common.variant_settings(settings, FOLDER, corpus_dir=str(corpus), redis_url=redis_url)
    (corpus / FOLDER).mkdir()
    for doc in docs:
        (corpus / path_of(doc)).write_bytes(document_bytes(doc))
    await common.reset_variant(settings, s)
    async with await AsyncConnection.connect(s.pg_dsn) as conn:
        await pg.insert_tuple(conn, args.user, "viewer", f"folder:{FOLDER}")
    try:
        initial, rows = await run_trials(s, args, docs)
    finally:
        await common.drop_variant(settings, s)
        shutil.rmtree(corpus, ignore_errors=True)
    found, gone = [r[0] for r in rows], [r[1] for r in rows]
    summary = {
        "p50": [percentile(found, 50), percentile(gone, 50)],
        "max": [max(found), max(gone)],
    }
    metrics = {"documents": len(docs), "initial_index_s": initial}
    metrics |= {"searchable_s": found, "evicted_s": gone, **summary}
    await common.save_run(settings, args, metrics)
    lines = [[i, *r] for i, r in enumerate(rows, start=1)]
    lines += [[name, *values] for name, values in summary.items()]
    footer = f"\n{len(docs)} documents first indexed in {initial} s (including model loading)"
    return common.table(["trial", "searchable_s", "evicted_s"], lines) + footer


def parser() -> argparse.ArgumentParser:
    description = "Seconds from a file change to searchable and to cache eviction."
    parser = common.parser(description, "freshness")
    parser.add_argument("--docs", type=int, default=20, help="documents in the watched folder")
    parser.add_argument("--trials", type=int, default=5, help="file changes to time")
    parser.add_argument("--redis-db", type=int, default=1, help="scratch Redis database, emptied")
    parser.add_argument("--poll-ms", type=int, default=50, help="how often to check")
    parser.add_argument("--timeout-s", type=float, default=60, help="give up on a trial after")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
