import uuid

import httpx
from test_ask import ask

from rag.api.pipeline import Deps


async def post_feedback(
    client: httpx.AsyncClient, api_key: str, request_id: str, rating: int = 1
) -> httpx.Response:
    return await client.post(
        "/feedback",
        json={"request_id": request_id, "rating": rating, "comment": "clear answer"},
        headers={"X-API-Key": api_key},
    )


async def test_feedback_is_stored(client: httpx.AsyncClient, api_deps: Deps) -> None:
    events = await ask(client, "dev-ana", "How many requests per minute does the public API allow?")
    request_id = events[-1][1]["request_id"]
    response = await post_feedback(client, "dev-ana", request_id, rating=-1)
    assert response.status_code == 204
    async with api_deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT rating, comment FROM feedback WHERE request_id = %s", (request_id,)
        )
        assert await cur.fetchall() == [(-1, "clear answer")]


async def test_unknown_request_id_returns_404(client: httpx.AsyncClient) -> None:
    response = await post_feedback(client, "dev-ana", str(uuid.uuid4()))
    assert response.status_code == 404
    assert response.json() == {"code": "not_found", "message": "unknown request_id"}


async def test_another_users_request_returns_404(client: httpx.AsyncClient) -> None:
    events = await ask(client, "dev-ana", "How many requests per minute does the public API allow?")
    response = await post_feedback(client, "dev-ben", events[-1][1]["request_id"])
    assert response.status_code == 404
