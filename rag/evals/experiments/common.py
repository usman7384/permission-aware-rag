import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff

from rag.api import pipeline
from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.ids import doc_id
from rag.common.migrate import migrate
from rag.common.models import LLM, FakeLLM, make_embedder
from rag.evals.benchmark import DATA_DIR, BenchDoc, bench_settings, load, read_queries
from rag.evals.score import NOT_CONFIG, Golden, benchmark_questions, metrics_of, run_question
from rag.evals.synthetic import synthetic_acl
from rag.worker import main as worker
from rag.worker.chunk import Counter

Points = list[tuple[dict[str, Any], list[float]]]
Experiment = Callable[[Settings, argparse.Namespace], Awaitable[str]]
# Sampled documents get one synthetic folder, so their paths differ from the full benchmark's.
SAMPLE_FOLDER = "sample/f0000"


THRESHOLD_HELP = (
    "Qdrant indexing and full-scan threshold (KB) of the copy; at Qdrant's default of 10000 "
    "a SciFact-sized collection is never HNSW-indexed and every search is a full scan"
)


def parser(description: str, name: str) -> argparse.ArgumentParser:
    """The arguments every experiment takes: where the benchmark is and how to score it."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--name", default=name, help="name of the eval_runs row")
    parser.add_argument("--benchmark", type=Path, default=DATA_DIR / "scifact", help="BEIR folder")
    parser.add_argument("--split", default="test", help="benchmark qrels split")
    parser.add_argument("--user", default="user:bench", help="benchmark user")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N queries")
    parser.add_argument("--database", default="rag_bench", help="benchmark Postgres database")
    parser.add_argument("--prefix", default="bench_", help="benchmark Qdrant/OpenSearch prefix")
    # The CPU cross-encoder needs more than the API's 2 s budget for 50 to 100 candidates.
    parser.add_argument("--rerank-timeout-ms", type=int, default=20000, help="reranker budget")
    return parser


def run_main(experiment: Experiment, args: argparse.Namespace) -> None:
    """Run the experiment on the benchmark stores and print its table."""
    settings = bench_settings(Settings(), args.database, args.prefix)
    settings = settings.model_copy(update={"rerank_timeout_ms": args.rerank_timeout_ms})
    table_text = asyncio.run(experiment(settings, args), loop_factory=asyncio.SelectorEventLoop)
    print(table_text)


async def save_run(settings: Settings, args: argparse.Namespace, metrics: dict[str, Any]) -> str:
    """Store one `eval_runs` row named `args.name`, with the arguments and settings as config."""
    run_id = str(uuid.uuid4())
    config = {
        "args": json.loads(json.dumps(vars(args), default=str)),
        "settings": settings.model_dump(exclude=NOT_CONFIG),
    }
    async with await AsyncConnection.connect(settings.pg_dsn) as conn:
        await pg.insert_eval_run(conn, run_id, args.name, config, metrics)
    return run_id


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Rows as right-aligned text columns, floats to three decimals."""
    cells = [headers] + [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    return "\n".join(
        "  ".join(c.rjust(w) for c, w in zip(row, widths, strict=True)) for row in cells
    )


async def labelled_questions(pool: AsyncConnectionPool, args: argparse.Namespace) -> list[Golden]:
    """The benchmark queries whose relevant documents are loaded, up to `--limit`."""
    async with pool.connection() as conn:
        paths = await pg.live_document_paths(conn)
    return benchmark_questions(args.benchmark, args.split, args.user, paths, args.limit)


def sample_corpus(
    docs: list[BenchDoc], qrels: dict[str, dict[str, int]], queries: int | None, size: int
) -> list[BenchDoc]:
    """The relevant documents of the first labelled queries, then other documents up to `size`,
    in corpus order."""
    labelled = [grades for grades in qrels.values() if any(g > 0 for g in grades.values())]
    present = {doc.id for doc in docs}
    wanted = {d for grades in labelled[:queries] for d, g in grades.items() if g > 0} & present
    for doc in docs:
        if len(wanted) >= size:
            break
        wanted.add(doc.id)
    return [doc for doc in docs if doc.id in wanted]


def variant_settings(base: Settings, variant: str, **changes: Any) -> Settings:
    """The base settings with their own database `{base}_{variant}` and INDEX_VERSION `variant`."""
    database = f"{conninfo_to_dict(base.pg_dsn)['dbname']}_{variant}"
    dsn = make_conninfo(base.pg_dsn, dbname=database)
    return base.model_copy(update={"pg_dsn": dsn, "index_version": variant, **changes})


async def drop_indexes(settings: Settings) -> None:
    """Delete the settings' Qdrant collections and OpenSearch index."""
    qdrant, opensearch = qdrant_.connect(settings), opensearch_.connect(settings)
    try:
        await drop_collections(qdrant, settings)
        await opensearch.indices.delete(index=settings.chunks_name, ignore_unavailable=True)
    finally:
        await qdrant.close()
        await opensearch.close()


async def drop_variant(base: Settings, settings: Settings) -> None:
    """Delete a variant's database, collections and index."""
    database = conninfo_to_dict(settings.pg_dsn)["dbname"]
    await pg.drop_database(base.pg_dsn, database, base.store_timeout_s)
    await drop_indexes(settings)


async def reset_variant(base: Settings, settings: Settings) -> None:
    """Drop the variant's stores, then create and migrate its database."""
    await drop_variant(base, settings)
    database = conninfo_to_dict(settings.pg_dsn)["dbname"]
    await pg.ensure_database(base.pg_dsn, database, base.store_timeout_s)
    await migrate(settings, job_id=f"experiment-{settings.index_version}")


async def build_variant(
    base: Settings, settings: Settings, docs: list[BenchDoc], user: str, llm: LLM | None = None
) -> dict[str, Any]:
    """Index the documents into the variant's emptied stores; returns the load counts."""
    await reset_variant(base, settings)
    acl = synthetic_acl(SAMPLE_FOLDER.split("/")[0], len(docs), 1, 1.0, user)
    deps = await worker.open_deps(settings)
    deps.llm = llm or deps.llm
    try:
        return await load(deps, docs, acl, user, batch_size=64)
    finally:
        await worker.close_deps(deps)


@dataclass
class CountingLLM:
    """An LLM that counts its calls and their prompt and completion tokens."""

    llm: LLM
    count: Counter
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        reply = await self.llm.complete(system, user, model=model, max_tokens=max_tokens)
        self.calls += 1
        self.prompt_tokens += self.count(system) + self.count(user)
        self.completion_tokens += self.count(reply)
        return reply

    def stream(self, system: str, user: str, *, model: str, max_tokens: int) -> AsyncIterator[str]:
        return self.llm.stream(system, user, model=model, max_tokens=max_tokens)


def fake_llm_unless_configured(settings: Settings) -> LLM | None:
    """The fake LLM when real models are used but no LLM endpoint is configured, else None."""
    return FakeLLM() if settings.model_backend == "real" and not settings.llm_base_url else None


def llm_name(settings: Settings) -> str:
    """The LLM an experiment's answers, judgements or contexts come from."""
    fake = settings.model_backend == "fake" or fake_llm_unless_configured(settings) is not None
    return "fake" if fake else str(settings.llm_model)


async def evaluate_variant(
    settings: Settings, args: argparse.Namespace, docs: list[BenchDoc], with_answers: bool
) -> dict[str, Any]:
    """Metrics of the labelled questions on one variant's stores."""
    deps = await pipeline.open_deps(settings)
    deps.llm = fake_llm_unless_configured(settings) or deps.llm
    # Every variant has the same document and parent IDs, so `chunk:` keys must not carry over.
    doc_ids = [str(doc_id(f"{SAMPLE_FOLDER}/{doc.id}.txt")) for doc in docs]
    try:
        await redis_.delete_parents_of(deps.redis, doc_ids)
        questions = await labelled_questions(deps.pg, args)
        results = [await run_question(deps, q, with_answers) for q in questions]
        await redis_.delete_parents_of(deps.redis, doc_ids)
    finally:
        await pipeline.close_deps(deps)
    return metrics_of(questions, results, with_answers)


async def build_collection(
    client: AsyncQdrantClient,
    settings: Settings,
    points: Points,
    threshold_kb: int,
    full_scan_kb: int | None = None,
) -> float:
    """Create the settings' chunks collection from the points, with Qdrant's indexing threshold
    set to `threshold_kb` and its full-scan threshold to `full_scan_kb` (default the same);
    returns seconds until Qdrant has finished optimizing it."""
    start = time.perf_counter()
    await drop_collections(client, settings)
    await qdrant_.ensure_collections(client, settings)
    await client.update_collection(
        settings.chunks_name,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=threshold_kb),
        hnsw_config=HnswConfigDiff(full_scan_threshold=full_scan_kb or threshold_kb),
    )
    for first in range(0, len(points), 1000):
        batch = points[first : first + 1000]
        await qdrant_.upsert_chunks(client, settings, [p for p, _ in batch], [v for _, v in batch])
    async with asyncio.timeout(settings.index_wait_timeout_s):
        while (await client.get_collection(settings.chunks_name)).status != "green":
            await asyncio.sleep(0.2)
    return round(time.perf_counter() - start, 2)


async def drop_collections(client: AsyncQdrantClient, settings: Settings) -> None:
    await client.delete_collection(settings.chunks_name)
    await client.delete_collection(settings.answer_cache_name)


def overlap(found: list[str], truth: list[str]) -> float:
    """The share of the true nearest neighbours that a search found."""
    return len(set(found) & set(truth)) / len(truth) if truth else 1.0


async def timed_searches(
    client: AsyncQdrantClient,
    settings: Settings,
    vectors: list[list[float]],
    principals: list[str],
    top_k: int,
    exact: bool,
) -> tuple[list[list[str]], list[float]]:
    """The chunk IDs each query vector finds, and the milliseconds each search took."""
    found, ms = [], []
    for vector in vectors:
        start = time.perf_counter()
        hits = await qdrant_.search_chunks(client, settings, vector, principals, top_k, exact)
        ms.append((time.perf_counter() - start) * 1000)
        found.append([hit["chunk_id"] for hit in hits])
    return found, ms


async def query_vectors(
    settings: Settings, args: argparse.Namespace, count: int
) -> list[list[float]]:
    """Embedded texts of the benchmark's first `count` queries."""
    texts = list(read_queries(args.benchmark).values())[:count]
    embedder = make_embedder(settings)
    return await asyncio.to_thread(lambda: [embedder.embed_query(text) for text in texts])
