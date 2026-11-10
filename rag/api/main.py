import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.log import log


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.settings = settings
    app.state.pg = await pg.open_pool(settings)
    app.state.redis = redis_.connect(settings)
    app.state.qdrant = qdrant_.connect(settings)
    app.state.opensearch = opensearch_.connect(settings)
    try:
        yield
    finally:
        await app.state.pg.close()
        await app.state.redis.aclose()
        await app.state.qdrant.close()
        await app.state.opensearch.close()


app = FastAPI(title="rag-laptop", lifespan=lifespan)


async def llm_reachable(settings: Settings) -> None:
    """Open and close a TCP connection to LLM_BASE_URL; never sends a completion."""
    if settings.model_backend == "fake":
        return
    if not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL is not set")
    url = urlsplit(settings.llm_base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    _, writer = await asyncio.open_connection(url.hostname, port)
    writer.close()
    await writer.wait_closed()


async def run_check(name: str, check: Awaitable[None], timeout_s: float, request_id: str) -> bool:
    try:
        await asyncio.wait_for(check, timeout_s)
    except Exception as exc:
        log("health_check_failed", request_id=request_id, check=name, error=repr(exc))
        return False
    return True


@app.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    state = request.app.state
    request_id = str(uuid.uuid4())
    checks = {
        "postgres": pg.ping(state.pg),
        "redis": redis_.ping(state.redis),
        "qdrant": qdrant_.ping(state.qdrant),
        "opensearch": opensearch_.ping(state.opensearch),
        "llm": llm_reachable(state.settings),
    }
    timeout_s = state.settings.store_timeout_s
    results = await asyncio.gather(
        *(run_check(name, check, timeout_s, request_id) for name, check in checks.items())
    )
    status = dict(zip(checks, results, strict=True))
    healthy = all(results)
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": status},
        status_code=200 if healthy else 503,
    )
