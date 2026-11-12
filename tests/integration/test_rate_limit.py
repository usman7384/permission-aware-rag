import asyncio
import time

import httpx
import pytest

from rag.api.pipeline import Deps
from rag.common.keys import api_key_hash, rate_limit_key

LIMIT = 3


async def post_ask(client: httpx.AsyncClient, api_key: str) -> httpx.Response:
    return await client.post(
        "/ask",
        json={"session_id": "s-rl", "question": "ZEBRA-7"},
        headers={"X-API-Key": api_key},
    )


async def test_request_over_the_limit_in_one_minute_returns_429(
    client: httpx.AsyncClient, api_deps: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "rate_limit_per_min", LIMIT)
    if time.time() % 60 > 50:
        await asyncio.sleep(60 - time.time() % 60)
    for _ in range(LIMIT):
        assert (await post_ask(client, "dev-ana")).status_code == 200
    response = await post_ask(client, "dev-ana")
    assert response.status_code == 429
    assert response.json() == {"code": "rate_limited", "message": "too many requests this minute"}
    assert (await post_ask(client, "dev-ben")).status_code == 200
    key = rate_limit_key(api_key_hash("dev-ana"), int(time.time()) // 60)
    assert await api_deps.redis.get(key) == str(LIMIT + 1)
    assert 0 < await api_deps.redis.ttl(key) <= api_deps.settings.rate_limit_ttl_s
