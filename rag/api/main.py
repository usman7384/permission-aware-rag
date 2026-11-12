import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from rag.api.auth import ApiError, authenticate
from rag.api.generate import Event
from rag.api.guard import input_guard
from rag.api.pipeline import Deps, ask, close_deps, expand, open_deps, timed
from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.log import log
from rag.common.permissions import change_tuple


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    deps = await open_deps(Settings())
    app.state.deps = deps
    try:
        yield
    finally:
        await close_deps(deps)


app = FastAPI(title="rag-laptop", lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"code": exc.code, "message": exc.message}, status_code=exc.status)


async def llm_reachable(settings: Settings) -> None:
    if settings.model_backend == "fake":
        return
    if not settings.llm_base_url:
        raise ValueError("LLM_BASE_URL is not set")
    url = urlsplit(settings.llm_base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    _, writer = await asyncio.open_connection(url.hostname, port)
    writer.close()
    await writer.wait_closed()


async def inference_ready(url: str, timeout_s: float) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=timeout_s) as client:
        response = await client.get("/healthz")
        response.raise_for_status()


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
    if deps.settings.inference_url:
        checks["inference"] = inference_ready(deps.settings.inference_url, timeout_s)
    results = await asyncio.gather(
        *(run_check(name, check, timeout_s, request_id) for name, check in checks.items())
    )
    status = dict(zip(checks, results, strict=True))
    healthy = all(results)
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "checks": status},
        status_code=200 if healthy else 503,
    )


class AskRequest(BaseModel):
    session_id: str
    question: str


async def server_sent_events(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    async for name, payload in events:
        yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@app.post("/ask")
async def ask_endpoint(
    body: AskRequest,
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    deps: Deps = request.app.state.deps
    request_id = str(uuid.uuid4())
    timings: dict[str, float] = {}
    async with deps.pg.connection() as conn:
        with timed(timings, "authenticate"):
            caller = await authenticate(conn, deps.redis, deps.settings, x_api_key)
        with timed(timings, "expand"):
            principals, fp = await expand(conn, caller.user_id)
    with timed(timings, "input_guard"):
        input_guard(body.question, deps.settings.max_question_chars)
    events = ask(deps, caller, principals, fp, body.session_id, body.question, request_id, timings)
    return StreamingResponse(server_sent_events(events), media_type="text/event-stream")


class Feedback(BaseModel):
    request_id: UUID
    rating: Literal[1, -1]
    comment: str | None = None


@app.post("/feedback", status_code=204)
async def feedback(
    body: Feedback,
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Response:
    deps: Deps = request.app.state.deps
    async with deps.pg.connection() as conn:
        caller = await authenticate(conn, deps.redis, deps.settings, x_api_key)
        stored = await pg.insert_feedback(
            conn, str(body.request_id), caller.user_id, body.rating, body.comment
        )
    if not stored:
        raise ApiError(404, "not_found", "unknown request_id")
    log("feedback", request_id=str(body.request_id), user_id=caller.user_id, rating=body.rating)
    return Response(status_code=204)


class TupleChange(BaseModel):
    op: Literal["add", "remove"]
    subject: str
    relation: Literal["member", "viewer", "parent"]
    object: str


@app.post("/admin/tuples", status_code=204)
async def admin_tuples(
    body: TupleChange,
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Response:
    deps: Deps = request.app.state.deps
    request_id = str(uuid.uuid4())
    async with deps.pg.connection() as conn:
        caller = await authenticate(conn, deps.redis, deps.settings, x_api_key)
        if not caller.is_admin:
            raise ApiError(401, "unauthorized", "an admin API key is required")
        await change_tuple(conn, body.op, body.subject, body.relation, body.object)
    log("tuple_changed", request_id=request_id, user_id=caller.user_id, **body.model_dump())
    return Response(status_code=204)


class FaultChange(BaseModel):
    component: Literal["dense", "sparse", "reranker", "llm"]
    mode: Literal["ok", "slow", "down"]


@app.post("/admin/faults", status_code=204)
async def admin_faults(
    body: FaultChange,
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Response:
    deps: Deps = request.app.state.deps
    async with deps.pg.connection() as conn:
        caller = await authenticate(conn, deps.redis, deps.settings, x_api_key)
    if not caller.is_admin:
        raise ApiError(401, "unauthorized", "an admin API key is required")
    await redis_.set_fault(deps.redis, body.component, body.mode)
    log("fault_set", request_id=str(uuid.uuid4()), user_id=caller.user_id, **body.model_dump())
    return Response(status_code=204)
