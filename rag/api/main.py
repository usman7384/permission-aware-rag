import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag.api.auth import ApiError, authenticate
from rag.api.pipeline import Deps, search
from rag.api.retrieve import RetrievalUnavailable
from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.log import log
from rag.common.models import make_embedder, make_reranker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    deps = Deps(
        settings=settings,
        pg=await pg.open_pool(settings),
        redis=redis_.connect(settings),
        qdrant=qdrant_.connect(settings),
        opensearch=opensearch_.connect(settings),
        embedder=make_embedder(settings),
        reranker=make_reranker(settings),
    )
    app.state.deps = deps
    try:
        yield
    finally:
        await deps.pg.close()
        await deps.redis.aclose()
        await deps.qdrant.close()
        await deps.opensearch.close()


app = FastAPI(title="rag-laptop", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"code": exc.code, "message": exc.message}, status_code=exc.status)


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
    deps: Deps = request.app.state.deps
    request_id = str(uuid.uuid4())
    checks = {
        "postgres": pg.ping(deps.pg),
        "redis": redis_.ping(deps.redis),
        "qdrant": qdrant_.ping(deps.qdrant),
        "opensearch": opensearch_.ping(deps.opensearch),
        "llm": llm_reachable(deps.settings),
    }
    timeout_s = deps.settings.store_timeout_s
    results = await asyncio.gather(
        *(run_check(name, check, timeout_s, request_id) for name, check in checks.items())
    )
    status = dict(zip(checks, results, strict=True))
    healthy = all(results)
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": status},
        status_code=200 if healthy else 503,
    )


class SearchRequest(BaseModel):
    question: str


@app.post("/search")
async def search_endpoint(
    body: SearchRequest,
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """Temporary (milestone 2 only): the parents retrieval returns for a question."""
    deps: Deps = request.app.state.deps
    request_id = str(uuid.uuid4())
    async with deps.pg.connection() as conn:
        caller = await authenticate(conn, x_api_key)
    try:
        parents, degraded = await search(deps, caller, body.question, request_id)
    except RetrievalUnavailable as exc:
        raise ApiError(503, "retrieval_unavailable", str(exc)) from exc
    return JSONResponse(
        {
            "request_id": request_id,
            "parents": [asdict(parent) for parent in parents],
            "degraded": degraded,
        }
    )
