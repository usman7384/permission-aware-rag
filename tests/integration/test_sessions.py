import httpx
import pytest
from test_ask import ask

from rag.api.pipeline import Deps

FIRST = "How many requests per minute does the public API allow?"
SECOND = "And for admin keys?"


def spy_rewrites(monkeypatch: pytest.MonkeyPatch, deps: Deps, fail: bool) -> list[str]:
    """Record the user text of every rewrite call; with `fail`, make those calls raise."""
    calls: list[str] = []
    complete = deps.llm.complete

    async def spy(system: str, user: str, **kwargs: object) -> str:
        if system.startswith("KIND: rewrite"):
            calls.append(user)
            if fail:
                raise RuntimeError("rewriter down")
        return await complete(system, user, **kwargs)

    monkeypatch.setattr(deps.llm, "complete", spy)
    return calls


async def rewritten(deps: Deps, request_id: str) -> str | None:
    async with deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT rewritten FROM requests WHERE request_id = %s", (request_id,)
        )
        (value,) = await cur.fetchone()
    return value


async def test_second_question_is_rewritten_using_the_first(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_rewrites(monkeypatch, api_deps, fail=False)
    first = await ask(client, "dev-ana", FIRST, session_id="s-rw")
    assert calls == []
    assert await rewritten(api_deps, first[-1][1]["request_id"]) is None
    second = await ask(client, "dev-ana", SECOND, session_id="s-rw")
    [user_text] = calls
    assert user_text.splitlines()[0] == f"user: {FIRST}"
    assert user_text.splitlines()[-1] == f"question: {SECOND}"
    done = second[-1][1]
    assert done["degraded"] == []
    assert await rewritten(api_deps, done["request_id"]) == f"{FIRST} {SECOND}"


async def test_another_users_turns_are_not_history(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_rewrites(monkeypatch, api_deps, fail=False)
    await ask(client, "dev-ana", FIRST, session_id="s-shared")
    await ask(client, "dev-ben", SECOND, session_id="s-shared")
    assert calls == []


async def test_rewriter_failure_uses_the_question_and_reports_rewrite(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_rewrites(monkeypatch, api_deps, fail=True)
    await ask(client, "dev-ana", FIRST, session_id="s-fail")
    events = await ask(client, "dev-ana", FIRST, session_id="s-fail")
    assert len(calls) == 1
    done = events[-1][1]
    assert done["degraded"] == ["rewrite"]
    assert done["cache_tier"] == "exact"
    assert await rewritten(api_deps, done["request_id"]) is None
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == "Based on the documents, the answer is in [1]."
