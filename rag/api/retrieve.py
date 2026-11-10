import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any

from opensearchpy import AsyncOpenSearch
from psycopg import AsyncConnection
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.common import faults, opensearch_, pg, qdrant_
from rag.common.config import Settings
from rag.common.log import log
from rag.common.permissions import readable_docs


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    parent_id: str
    doc_id: str
    version: int
    score: float = 0.0


class RetrievalUnavailable(Exception):
    """Both retrieval branches failed."""


def _candidate(payload: dict[str, Any]) -> Candidate:
    return Candidate(
        chunk_id=payload["chunk_id"],
        parent_id=payload["parent_id"],
        doc_id=payload["doc_id"],
        version=payload["version"],
    )


async def dense(
    client: AsyncQdrantClient,
    redis: Redis,
    settings: Settings,
    vector: list[float],
    principals: list[str],
) -> list[Candidate]:
    await faults.inject(redis, "dense", settings.fault_slow_ms)
    payloads = await qdrant_.search_chunks(
        client, settings, vector, principals, settings.dense_top_k
    )
    return [_candidate(p) for p in payloads]


async def sparse(
    client: AsyncOpenSearch, redis: Redis, settings: Settings, question: str, principals: list[str]
) -> list[Candidate]:
    await faults.inject(redis, "sparse", settings.fault_slow_ms)
    payloads = await opensearch_.search_chunks(
        client, settings, question, principals, settings.sparse_top_k
    )
    return [_candidate(p) for p in payloads]


async def _branch(
    name: str, call: Awaitable[list[Candidate]], timeout_s: float, request_id: str
) -> list[Candidate] | None:
    """The branch's results, or None if it failed or timed out."""
    try:
        return await asyncio.wait_for(call, timeout_s)
    except Exception as exc:
        log("retrieval_branch_failed", request_id=request_id, branch=name, error=repr(exc))
        return None


async def retrieve(
    qdrant: AsyncQdrantClient,
    opensearch: AsyncOpenSearch,
    redis: Redis,
    settings: Settings,
    question: str,
    vector: list[float],
    principals: list[str],
    request_id: str,
) -> tuple[list[list[Candidate]], list[str]]:
    """Run both branches concurrently; returns the lists that succeeded and the failed names."""
    timeout_s = settings.retrieve_timeout_ms / 1000
    dense_call = dense(qdrant, redis, settings, vector, principals)
    sparse_call = sparse(opensearch, redis, settings, question, principals)
    results = await asyncio.gather(
        _branch("dense", dense_call, timeout_s, request_id),
        _branch("sparse", sparse_call, timeout_s, request_id),
    )
    failed = [name for name, r in zip(("dense", "sparse"), results, strict=True) if r is None]
    if len(failed) == len(results):
        raise RetrievalUnavailable("dense and sparse retrieval both failed")
    return [r for r in results if r is not None], failed


def rrf(branches: list[list[Candidate]], k: int, keep: int) -> list[Candidate]:
    """Reciprocal rank fusion: sum 1 / (k + rank) per branch, ties broken by chunk_id."""
    scores: dict[str, float] = {}
    chunks: dict[str, Candidate] = {}
    for branch in branches:
        for rank, candidate in enumerate(branch, start=1):
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1 / (k + rank)
            chunks[candidate.chunk_id] = candidate
    order = sorted(scores, key=lambda chunk: (-scores[chunk], chunk))
    return [replace(chunks[chunk], score=scores[chunk]) for chunk in order[:keep]]


async def recheck(
    conn: AsyncConnection, candidates: list[Candidate], principals: list[str]
) -> list[Candidate]:
    """Drop candidates that are deleted, of a stale version, or no longer readable."""
    doc_ids = sorted({c.doc_id for c in candidates})
    if not doc_ids:
        return []
    versions = await pg.live_document_versions(conn, doc_ids)
    readable = await readable_docs(conn, principals, doc_ids)
    return [c for c in candidates if c.doc_id in readable and versions.get(c.doc_id) == c.version]
