import httpx
import pytest
from test_ask import Events, ask
from test_cache import QUESTION, answer_events, done

from rag.api.pipeline import Deps

COMPONENTS = ("dense", "sparse", "reranker", "llm")
FAKE_ANSWER = "Based on the documents, the answer is in [1]."
FALLBACK = (
    "This document describes how the public API protects itself from overload and how client "
    "teams should plan their traffic. The public API allows 120 requests per minute per key."
)


async def set_fault(client: httpx.AsyncClient, component: str, mode: str) -> None:
    response = await client.post(
        "/admin/faults",
        json={"component": component, "mode": mode},
        headers={"X-API-Key": "dev-admin"},
    )
    assert response.status_code == 204


@pytest.fixture
def short_timeouts(api_deps: Deps, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_deps.settings, "rerank_timeout_ms", 200)
    monkeypatch.setattr(api_deps.settings, "llm_timeout_s", 0.2)
    s = api_deps.settings
    assert s.fault_slow_ms > max(s.retrieve_timeout_ms, s.rerank_timeout_ms, s.llm_timeout_s * 1000)


async def assert_not_cached(deps: Deps, events: Events) -> None:
    assert done(events)["cache_tier"] == "none"
    assert await deps.redis.keys("ans:*") == []


@pytest.mark.usefixtures("short_timeouts")
@pytest.mark.parametrize("mode", ["down", "slow"])
@pytest.mark.parametrize("branch", ["dense", "sparse"])
async def test_one_retrieval_branch_failing_answers_from_the_other(
    client: httpx.AsyncClient, api_deps: Deps, branch: str, mode: str
) -> None:
    await set_fault(client, branch, mode)
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["degraded"] == [branch]
    text, citations = answer_events(events)
    assert text == FAKE_ANSWER and citations[0]["path"] == "specs/api-limits.md"
    await assert_not_cached(api_deps, events)


async def test_both_retrieval_branches_down_is_retrieval_unavailable(
    client: httpx.AsyncClient, api_deps: Deps
) -> None:
    await set_fault(client, "dense", "down")
    await set_fault(client, "sparse", "down")
    events = await ask(client, "dev-ana", QUESTION)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["code"] == "retrieval_unavailable"
    assert await api_deps.redis.keys("ans:*") == []


@pytest.mark.usefixtures("short_timeouts")
@pytest.mark.parametrize("mode", ["down", "slow"])
async def test_reranker_failing_answers_in_fused_order(
    client: httpx.AsyncClient, api_deps: Deps, mode: str
) -> None:
    await set_fault(client, "reranker", mode)
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["degraded"] == ["rerank"]
    text, citations = answer_events(events)
    assert text == FAKE_ANSWER and len(citations) == 1
    await assert_not_cached(api_deps, events)


@pytest.mark.usefixtures("short_timeouts")
@pytest.mark.parametrize("mode", ["down", "slow"])
async def test_llm_failing_returns_the_extractive_fallback(
    client: httpx.AsyncClient, api_deps: Deps, mode: str
) -> None:
    await set_fault(client, "llm", mode)
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["degraded"] == ["llm"]
    text, citations = answer_events(events)
    lines = text.splitlines()
    assert 1 <= len(lines) == len(citations) <= 3
    markers = {f"[{item['n']}]": item["path"] for item in citations}
    assert all(line.rsplit(" ", 1)[-1] in markers for line in lines)
    top_marker = lines[0].rsplit(" ", 1)[-1]
    assert markers[top_marker] == "specs/api-limits.md"
    assert lines[0] == f"{FALLBACK} {top_marker}"
    await assert_not_cached(api_deps, events)


async def test_setting_a_fault_back_to_ok_restores_normal_answers(
    client: httpx.AsyncClient, api_deps: Deps
) -> None:
    for component in COMPONENTS:
        await set_fault(client, component, "down")
    assert [name for name, _ in await ask(client, "dev-ana", QUESTION)] == ["error"]
    for component in COMPONENTS:
        await set_fault(client, component, "ok")
    assert await api_deps.redis.keys("fault:*") == []
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["degraded"] == []
    assert answer_events(events)[0] == FAKE_ANSWER


async def test_a_non_admin_key_gets_401_from_admin_faults(
    client: httpx.AsyncClient, api_deps: Deps
) -> None:
    response = await client.post(
        "/admin/faults",
        json={"component": "llm", "mode": "down"},
        headers={"X-API-Key": "dev-ana"},
    )
    assert response.status_code == 401
    assert await api_deps.redis.keys("fault:*") == []
