from typing import Any

import httpx
import pytest

from rag.api.client import parse_sse
from rag.api.pipeline import Deps

Events = list[tuple[str, dict[str, Any]]]


async def ask(client: httpx.AsyncClient, api_key: str, question: str) -> Events:
    response = await client.post(
        "/ask",
        json={"session_id": "s-1", "question": question},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return list(parse_sse(response.text.splitlines()))


async def test_fixture_question_streams_tokens_then_citations_then_done(
    client: httpx.AsyncClient, api_deps: Deps
) -> None:
    events = await ask(client, "dev-ana", "How many requests per minute does the public API allow?")
    names = [name for name, _ in events]
    tokens = names.index("citations")
    assert tokens > 0 and set(names[:tokens]) == {"token"}
    assert names[tokens:] == ["citations", "done"]
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == "Based on the documents, the answer is in [1]."
    [item] = events[tokens][1]["items"]
    assert item["n"] == 1 and item["path"] == "specs/api-limits.md"
    done = events[-1][1]
    assert done["cache_tier"] == "none" and done["degraded"] == [] and not done["abstained"]
    async with api_deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT answer, abstained, %s::uuid = ANY(doc_ids) FROM requests WHERE request_id = %s",
            (item["doc_id"], done["request_id"]),
        )
        assert await cur.fetchone() == (answer, False, True)
        cur = await conn.execute("SELECT role, text FROM turns WHERE session_id = 's-1'")
        assert [role for role, _ in await cur.fetchall()] == ["user", "assistant"]


async def test_code_word_ana_cannot_read_abstains_without_calling_the_llm(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_llm(*args: object, **kwargs: object) -> None:
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(api_deps.llm, "stream", no_llm)
    monkeypatch.setattr(api_deps.llm, "complete", no_llm)
    events = await ask(client, "dev-ana", "MAPLE-4")
    assert events[0] == ("token", {"text": "I don't know based on the documents I can access."})
    assert events[1:-1] == [("citations", {"items": []})]
    assert events[-1][0] == "done" and events[-1][1]["abstained"] is True


async def test_bad_api_key_returns_401_as_plain_json(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/ask",
        json={"session_id": "s-1", "question": "ZEBRA-7"},
        headers={"X-API-Key": "dev-nobody"},
    )
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"code": "unauthorized", "message": "unknown API key"}
