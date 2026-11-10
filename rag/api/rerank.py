import asyncio
from dataclasses import replace

from opensearchpy import AsyncOpenSearch
from redis.asyncio import Redis

from rag.api.retrieve import Candidate
from rag.common import faults, opensearch_
from rag.common.config import Settings
from rag.common.log import log
from rag.common.models import Reranker


async def _scores(
    opensearch: AsyncOpenSearch,
    redis: Redis,
    reranker: Reranker,
    settings: Settings,
    question: str,
    candidates: list[Candidate],
) -> list[float]:
    await faults.inject(redis, "reranker", settings.fault_slow_ms)
    texts = await opensearch_.get_texts(opensearch, settings, [c.chunk_id for c in candidates])
    # A chunk missing here was removed by a re-index since retrieval; hydrate drops it.
    passages = [texts.get(c.chunk_id, "") for c in candidates]
    return await asyncio.to_thread(reranker.score, question, passages)


async def rerank(
    opensearch: AsyncOpenSearch,
    redis: Redis,
    reranker: Reranker,
    settings: Settings,
    question: str,
    candidates: list[Candidate],
    request_id: str,
) -> tuple[list[Candidate], bool]:
    """Top `RERANK_KEEP` by reranker score and whether that worked; on failure, the fused order."""
    if not candidates:
        return [], True
    try:
        scores = await asyncio.wait_for(
            _scores(opensearch, redis, reranker, settings, question, candidates),
            settings.rerank_timeout_ms / 1000,
        )
    except Exception as exc:
        log("rerank_failed", request_id=request_id, error=repr(exc))
        return candidates[: settings.rerank_keep], False
    ranked = sorted(zip(scores, candidates, strict=True), key=lambda pair: -pair[0])
    return [replace(c, score=s) for s, c in ranked[: settings.rerank_keep]], True
