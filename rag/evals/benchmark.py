import argparse
import asyncio
import csv
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import httpx
from psycopg.conninfo import make_conninfo

from rag.common import opensearch_, pg, qdrant_
from rag.common.config import Settings
from rag.common.ids import doc_id
from rag.common.log import log
from rag.common.migrate import migrate
from rag.evals.synthetic import Acl, reader_group, synthetic_acl
from rag.worker.chunk import Parent, chunk
from rag.worker.index import Deps, chunk_payloads, cleanup, commit, emit, hash_file, stamp
from rag.worker.main import close_deps, open_deps
from rag.worker.parse import parse

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = Path(__file__).parents[2] / "data" / "beir"


@dataclass(frozen=True)
class BenchDoc:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Pending:
    doc: UUID
    path: str
    content_hash: str
    version: int
    parents: list[Parent]
    principals: list[str]


def download(name: str, data_dir: Path, timeout_s: float) -> Path:
    """The BEIR dataset's folder, downloaded and unzipped from the BEIR site if missing."""
    folder = data_dir / name
    if (folder / "corpus.jsonl").is_file():
        return folder
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    with httpx.stream("GET", BEIR_URL.format(name=name), timeout=timeout_s) as response:
        response.raise_for_status()
        with archive.open("wb") as out:
            for block in response.iter_bytes():
                out.write(block)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    archive.unlink()
    return folder


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_corpus(folder: Path, limit: int | None = None) -> list[BenchDoc]:
    rows = _jsonl(folder / "corpus.jsonl")[:limit]
    return [BenchDoc(str(r["_id"]), r.get("title", ""), r["text"]) for r in rows]


def read_queries(folder: Path) -> dict[str, str]:
    return {str(r["_id"]): r["text"] for r in _jsonl(folder / "queries.jsonl")}


def read_qrels(folder: Path, split: str) -> dict[str, dict[str, int]]:
    """Relevance grade by query id, then document id; grade 0 means judged not relevant."""
    qrels: dict[str, dict[str, int]] = {}
    with (folder / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def document_bytes(doc: BenchDoc) -> bytes:
    return (f"{doc.title}\n\n{doc.text}" if doc.title else doc.text).encode("utf-8")


def bench_name(path: str) -> str:
    """The benchmark document id that a loaded document's path was built from."""
    return PurePosixPath(path).stem


def bench_settings(settings: Settings, database: str, prefix: str) -> Settings:
    """The settings pointed at a separate Postgres database and Qdrant/OpenSearch prefix."""
    return settings.model_copy(
        update={"pg_dsn": make_conninfo(settings.pg_dsn, dbname=database), "store_prefix": prefix}
    )


async def prepare(deps: Deps, path: str, data: bytes) -> Pending | None:
    """Parse, chunk and stamp one document; None when it is already indexed unchanged."""
    doc, content_hash, s = doc_id(path), hash_file(data), deps.settings
    async with deps.pg.connection() as conn:
        row = await pg.get_document(conn, doc)
        if row is not None and row[0] == content_hash and not row[2]:
            return None
        principals = await stamp(conn, path, doc)
    parents = chunk(parse(path, data), s.parent_tokens, s.child_tokens, deps.embedder.count_tokens)
    return Pending(doc, path, content_hash, row[1] + 1 if row else 1, parents, principals)


async def index_batch(deps: Deps, batch: list[Pending]) -> int:
    """Index a batch with one embedding call and one write per store; returns the chunk count."""
    s, payloads, texts = deps.settings, [], []
    for item in batch:
        contexts: list[str | None] = [None] * len(item.parents)
        p, t = chunk_payloads(item.doc, item.version, item.parents, contexts, item.principals)
        payloads += p
        texts += t
    if payloads:
        vectors = await asyncio.to_thread(deps.embedder.embed_documents, texts)
        await qdrant_.upsert_chunks(deps.qdrant, s, payloads, vectors)
        await opensearch_.index_chunks(deps.opensearch, s, payloads, texts)
    for item in batch:
        contexts = [None] * len(item.parents)
        version, parents = item.version, item.parents
        await commit(deps, item.doc, item.path, item.content_hash, version, parents, contexts)
        # A first version has nothing older to clean up and no cached answer to purge.
        if item.version > 1:
            await cleanup(deps, item.doc, item.version)
            await emit(deps, "doc_changed", item.doc)
    return len(payloads)


async def write_acl(deps: Deps, acl: Acl, user: str) -> None:
    """Insert the viewer tuples and replace the reader group's memberships."""
    async with deps.pg.connection() as conn, conn.transaction():
        await pg.delete_subject_tuples(conn, reader_group(user), "member")
        for subject, relation, obj in acl.viewer_tuples + acl.member_tuples:
            await pg.insert_tuple(conn, subject, relation, obj)


async def load(
    deps: Deps, docs: list[BenchDoc], acl: Acl, user: str, batch_size: int
) -> dict[str, Any]:
    """Write the documents through the index functions in batches; returns counts and seconds."""
    start = time.perf_counter()
    paths = [f"{folder}/{doc.id}.txt" for folder, doc in zip(acl.folders, docs, strict=True)]
    async with deps.pg.connection() as conn:
        others = set(await pg.live_document_paths(conn)) - set(paths)
    if others:
        raise ValueError(f"{len(others)} other documents are loaded; use a fresh database")
    await write_acl(deps, acl, user)
    items, indexed, chunks = list(zip(paths, docs, strict=True)), 0, 0
    for first in range(0, len(items), batch_size):
        slice_ = items[first : first + batch_size]
        pending = [await prepare(deps, path, document_bytes(doc)) for path, doc in slice_]
        batch = [item for item in pending if item is not None]
        chunks += await index_batch(deps, batch)
        indexed += len(batch)
        log("benchmark_batch", job_id="benchmark", done=first + len(slice_), of=len(items))
    seconds = round(time.perf_counter() - start, 1)
    return {"documents": len(docs), "indexed": indexed, "chunks": chunks, "seconds": seconds}


async def run(args: argparse.Namespace) -> None:
    base = Settings()
    folder = download(args.dataset, args.data_dir, base.download_timeout_s)
    docs = read_corpus(folder, args.limit)
    acl = synthetic_acl(args.dataset, len(docs), args.folders, args.selectivity, args.user)
    settings = bench_settings(base, args.database, args.prefix)
    await pg.ensure_database(base.pg_dsn, args.database, base.store_timeout_s)
    await migrate(settings, job_id="benchmark")
    deps = await open_deps(settings)
    try:
        stats = await load(deps, docs, acl, args.user, args.batch_size)
    finally:
        await close_deps(deps)
    stats["selectivity"] = round(acl.selectivity, 6)
    for key, value in stats.items():
        print(f"{key:<12}{value:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a BEIR benchmark into separate stores.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="download folder")
    parser.add_argument("--limit", type=int, default=None, help="load only the first N documents")
    parser.add_argument("--batch-size", type=int, default=64, help="documents per write")
    parser.add_argument("--folders", type=int, default=1, help="synthetic folders")
    parser.add_argument("--selectivity", type=float, default=1.0, help="share --user can read")
    parser.add_argument("--user", default="user:bench", help="user the tuples are made for")
    parser.add_argument("--database", default="rag_bench", help="Postgres database")
    parser.add_argument("--prefix", default="bench_", help="Qdrant and OpenSearch name prefix")
    asyncio.run(run(parser.parse_args()), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
