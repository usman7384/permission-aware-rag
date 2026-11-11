import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_ask import Events, ask
from test_tuple_changes import run_invalidator

from rag.api import pipeline
from rag.common import qdrant_, redis_
from rag.common.ids import answer_point_id, doc_id
from rag.common.keys import cache_key
from rag.worker.index import Deps
from rag.worker.main import consume_once

QUESTION = "How many requests per minute does the public API allow?"
REWORDED = "How many requests per minute does the public API permit?"
REWORDED_NUMBER = "How many requests per minute does the public API permit for 2 keys?"


def done(events: Events) -> dict[str, Any]:
    name, payload = events[-1]
    assert name == "done"
    return payload


def answer_events(events: Events) -> tuple[str, list[dict[str, Any]]]:
    text = "".join(payload["text"] for name, payload in events if name == "token")
    [citations] = [payload["items"] for name, payload in events if name == "citations"]
    return text, citations


async def key_for(deps: pipeline.Deps, user: str, question: str) -> str:
    async with deps.pg.connection() as conn:
        _, fp = await pipeline.expand(conn, user)
    return cache_key(question, fp, deps.settings.index_version)


async def answer_points(deps: pipeline.Deps) -> int:
    return (await deps.qdrant.count(deps.settings.answer_cache_name)).count


async def point_exists(deps: pipeline.Deps, key: str) -> bool:
    found = await deps.qdrant.retrieve(deps.settings.answer_cache_name, [str(answer_point_id(key))])
    return bool(found)


async def cached_parent_docs(deps: pipeline.Deps) -> set[str]:
    """The documents of the parents held in `chunk:` keys."""
    keys = [key async for key in deps.redis.scan_iter(match="chunk:*")]
    return {json.loads(value)["doc_id"] for value in await deps.redis.mget(keys)} if keys else set()


def forbid(monkeypatch: pytest.MonkeyPatch, deps: pipeline.Deps) -> None:
    """Make retrieval and the LLM fail the test if they are called."""

    def called(*args: object, **kwargs: object) -> None:
        raise AssertionError("retrieval and the LLM must not be called on a cache hit")

    monkeypatch.setattr(pipeline, "retrieve", called)
    monkeypatch.setattr(deps.llm, "stream", called)
    monkeypatch.setattr(deps.llm, "complete", called)


async def test_second_identical_question_is_served_from_the_exact_cache(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await ask(client, "dev-ana", QUESTION)
    assert done(first)["cache_tier"] == "none"
    forbid(monkeypatch, api_deps)
    second = await ask(client, "dev-ana", QUESTION)
    assert done(second) | {"request_id": ""} == {
        "request_id": "",
        "cache_tier": "exact",
        "degraded": [],
        "abstained": False,
    }
    assert answer_events(second) == answer_events(first)
    async with api_deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT cache_tier, timings ? 'retrieve' FROM requests WHERE request_id = %s",
            (done(second)["request_id"],),
        )
        assert await cur.fetchone() == ("exact", False)


async def test_case_and_trailing_punctuation_hit_the_exact_cache(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    await ask(client, "dev-ana", QUESTION)
    forbid(monkeypatch, api_deps)
    events = await ask(
        client, "dev-ana", "  how many REQUESTS per minute does the public api allow"
    )
    assert done(events)["cache_tier"] == "exact"


async def test_reworded_question_hits_the_semantic_cache_unless_a_number_differs(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "semantic_cache_threshold", 0.8)
    first = await ask(client, "dev-ana", QUESTION)
    reworded = await ask(client, "dev-ana", REWORDED)
    assert done(reworded)["cache_tier"] == "semantic"
    assert answer_events(reworded) == answer_events(first)
    with_number = await ask(client, "dev-ana", REWORDED_NUMBER)
    assert done(with_number)["cache_tier"] == "none"


async def test_one_users_cached_answer_is_never_served_to_another(
    client: httpx.AsyncClient, api_deps: pipeline.Deps
) -> None:
    await ask(client, "dev-ana", QUESTION)
    assert done(await ask(client, "dev-ana", QUESTION))["cache_tier"] == "exact"
    assert done(await ask(client, "dev-ben", QUESTION))["cache_tier"] == "none"
    assert done(await ask(client, "dev-ben", QUESTION))["cache_tier"] == "exact"
    assert await key_for(api_deps, "user:ana", QUESTION) != await key_for(
        api_deps, "user:ben", QUESTION
    )


async def test_editing_a_context_document_evicts_its_cached_answer(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    await run_invalidator(ingested)  # drain the events of the initial ingestion
    first = await ask(client, "dev-ana", QUESTION)
    key = await key_for(api_deps, "user:ana", QUESTION)
    edited = str(doc_id("specs/api-limits.md"))
    cached = await redis_.get_answer(api_deps.redis, key)
    assert cached is not None and edited in cached["doc_ids"]
    assert await api_deps.redis.sismember(f"doc2ans:{edited}", key)
    assert await point_exists(api_deps, key)
    parent_docs = await cached_parent_docs(api_deps)
    assert edited in parent_docs

    file = Path(ingested.settings.corpus_dir) / "specs" / "api-limits.md"
    file.write_text(file.read_text(encoding="utf-8") + "\nBursts are smoothed.\n", "utf-8")
    await redis_.enqueue_job(ingested.redis, "upsert", "specs/api-limits.md")
    while await consume_once(ingested, "test"):
        pass
    await run_invalidator(ingested)

    assert await redis_.get_answer(api_deps.redis, key) is None
    assert not await api_deps.redis.exists(f"doc2ans:{edited}")
    assert not await point_exists(api_deps, key)
    assert await cached_parent_docs(api_deps) == parent_docs - {edited}
    assert done(await ask(client, "dev-ana", QUESTION))["cache_tier"] == "none"
    assert done(first)["cache_tier"] == "none"


async def test_abstentions_and_degraded_answers_are_not_cached(
    client: httpx.AsyncClient, api_deps: pipeline.Deps
) -> None:
    for _ in range(2):
        events = await ask(client, "dev-ana", "MAPLE-4")
        assert done(events)["abstained"] and done(events)["cache_tier"] == "none"
    await redis_.set_fault(api_deps.redis, "reranker", "down")
    for _ in range(2):
        events = await ask(client, "dev-ana", QUESTION)
        assert done(events)["degraded"] == ["rerank"] and done(events)["cache_tier"] == "none"
    assert await api_deps.redis.keys("ans:*") == []
    assert await api_deps.redis.keys("doc2ans:*") == []
    assert await answer_points(api_deps) == 0


async def test_expired_answer_behind_a_semantic_hit_is_a_miss_and_drops_the_point(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "semantic_cache_threshold", 0.8)
    await ask(client, "dev-ana", QUESTION)
    key = await key_for(api_deps, "user:ana", QUESTION)
    assert await point_exists(api_deps, key)
    await api_deps.redis.delete(f"ans:{key}")
    events = await ask(client, "dev-ana", REWORDED)
    assert done(events)["cache_tier"] == "none"
    assert not await point_exists(api_deps, key)


async def test_answer_cached_before_a_document_changed_is_not_served(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    await ask(client, "dev-ana", QUESTION)
    key = await key_for(api_deps, "user:ana", QUESTION)
    edited = str(doc_id("specs/api-limits.md"))
    assert (await redis_.get_answer(api_deps.redis, key))["doc_versions"][edited] == 1
    file = Path(ingested.settings.corpus_dir) / "specs" / "api-limits.md"
    file.write_text(file.read_text(encoding="utf-8") + "\nBursts are smoothed.\n", "utf-8")
    await redis_.enqueue_job(ingested.redis, "upsert", "specs/api-limits.md")
    while await consume_once(ingested, "test"):
        pass
    # The invalidator does not run, so the stale entry is still in the cache.
    assert await redis_.get_answer(api_deps.redis, key) is not None
    assert done(await ask(client, "dev-ana", QUESTION))["cache_tier"] == "none"
    assert (await redis_.get_answer(api_deps.redis, key))["doc_versions"][edited] == 2


async def test_answer_cached_before_access_was_revoked_is_not_served(
    client: httpx.AsyncClient, api_deps: pipeline.Deps
) -> None:
    assert done(await ask(client, "dev-ben", QUESTION))["cache_tier"] == "none"
    key = await key_for(api_deps, "user:ben", QUESTION)
    assert await redis_.get_answer(api_deps.redis, key) is not None
    limits = f"doc:{doc_id('specs/api-limits.md')}"
    body = {"op": "remove", "subject": "user:ben", "relation": "viewer", "object": limits}
    response = await client.post("/admin/tuples", json=body, headers={"X-API-Key": "dev-admin"})
    assert response.status_code == 204
    events = await ask(client, "dev-ben", QUESTION)
    assert done(events)["cache_tier"] == "none"
    assert all(c["path"] != "specs/api-limits.md" for c in answer_events(events)[1])
    # The fresh answer, from the documents ben can still read, replaced the cached one.
    cached = await redis_.get_answer(api_deps.redis, key)
    assert cached is None or str(doc_id("specs/api-limits.md")) not in cached["doc_versions"]


async def test_negated_rewording_does_not_hit_the_semantic_cache(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "semantic_cache_threshold", 0.8)
    negated = "How many requests per minute does the public API not allow?"
    await ask(client, "dev-ana", QUESTION)
    async with api_deps.pg.connection() as conn:
        _, fp = await pipeline.expand(conn, "user:ana")
    vector = await pipeline.embed(api_deps.embedder, negated)
    nearest = await qdrant_.nearest_answer(api_deps.qdrant, api_deps.settings, vector, fp)
    assert nearest is not None and nearest[1] >= 0.8
    assert done(await ask(client, "dev-ana", negated))["cache_tier"] == "none"


async def test_concurrent_identical_misses_run_the_pipeline_once(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = 0
    search = pipeline.search

    async def slow_search(*args: Any, **kwargs: Any) -> Any:
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.3)
        return await search(*args, **kwargs)

    monkeypatch.setattr(pipeline, "search", slow_search)
    results = await asyncio.gather(*(ask(client, "dev-ana", QUESTION) for _ in range(5)))
    tiers = sorted(done(events)["cache_tier"] for events in results)
    assert runs == 1 and tiers == ["exact"] * 4 + ["none"]
    assert len({answer_events(events)[0] for events in results}) == 1
    assert api_deps.inflight == {}
