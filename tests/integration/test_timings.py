import httpx
import pytest
from test_ask import ask
from test_cache import QUESTION, REWORDED, done

from rag.api.pipeline import Deps

STEPS = {
    "authenticate",
    "expand",
    "input_guard",
    "rewrite",
    "exact_cache",
    "embed",
    "semantic_cache",
    "retrieve",
    "rrf",
    "recheck",
    "rerank",
    "hydrate",
    "assemble",
    "generate",
    "output_guard",
    "persist",
}
EXACT_HIT = {"authenticate", "expand", "input_guard", "rewrite", "exact_cache", "persist"}


async def stored_timings(deps: Deps, request_id: str) -> dict[str, float]:
    async with deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT timings FROM requests WHERE request_id = %s", (request_id,)
        )
        (timings,) = await cur.fetchone()
    assert all(isinstance(ms, float | int) and ms >= 0 for ms in timings.values())
    return timings


async def test_a_cache_miss_times_all_sixteen_steps(
    client: httpx.AsyncClient, api_deps: Deps
) -> None:
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["cache_tier"] == "none"
    assert set(await stored_timings(api_deps, done(events)["request_id"])) == STEPS


async def test_cache_hits_time_only_the_steps_that_ran(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "semantic_cache_threshold", 0.8)
    await ask(client, "dev-ana", QUESTION)
    exact = await ask(client, "dev-ana", QUESTION)
    assert done(exact)["cache_tier"] == "exact"
    assert set(await stored_timings(api_deps, done(exact)["request_id"])) == EXACT_HIT
    semantic = await ask(client, "dev-ana", REWORDED)
    assert done(semantic)["cache_tier"] == "semantic"
    semantic_steps = EXACT_HIT | {"embed", "semantic_cache"}
    assert set(await stored_timings(api_deps, done(semantic)["request_id"])) == semantic_steps
