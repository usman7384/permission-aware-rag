import httpx

from rag.api.main import app
from rag.common.config import Settings


async def test_all_checks_true_with_stores_up(settings: Settings) -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {
            "postgres": True,
            "redis": True,
            "qdrant": True,
            "opensearch": True,
            "llm": True,
        },
    }
