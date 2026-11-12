import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from rag.common.config import Settings
from rag.common.models import Embedder, Reranker, local_embedder, local_reranker


class Gate:
    def __init__(self, concurrency: int, queue_max: int) -> None:
        self.slots = asyncio.Semaphore(concurrency)
        self.limit = concurrency + queue_max
        self.admitted = 0
        self.calls = 0
        self.rejected = 0

    @asynccontextmanager
    async def enter(self) -> AsyncIterator[None]:
        self.calls += 1
        if self.admitted >= self.limit:
            self.rejected += 1
            raise HTTPException(503, "inference queue is full")
        self.admitted += 1
        try:
            async with self.slots:
                yield
        finally:
            self.admitted -= 1


async def run_model[T](request: Request, call: Callable[..., T], *args: object) -> T:
    async with request.app.state.gate.enter():
        return await asyncio.to_thread(call, *args)


class EmbedRequest(BaseModel):
    texts: list[str]
    kind: Literal["query", "document"]


class RerankRequest(BaseModel):
    query: str
    passages: list[str]


def create_app(settings: Settings, embedder: Embedder, reranker: Reranker) -> FastAPI:
    app = FastAPI(title="permission-aware-rag inference")

    def embed_queries(texts: list[str]) -> list[list[float]]:
        return [embedder.embed_query(text) for text in texts]

    app.state.gate = Gate(settings.inference_concurrency, settings.inference_queue_max)

    @app.post("/embed")
    async def embed(body: EmbedRequest, request: Request) -> dict[str, list[list[float]]]:
        call = embedder.embed_documents if body.kind == "document" else embed_queries
        return {"vectors": await run_model(request, call, body.texts)}

    @app.post("/rerank")
    async def rerank(body: RerankRequest, request: Request) -> dict[str, list[float]]:
        return {"scores": await run_model(request, reranker.score, body.query, body.passages)}

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, object]:
        gate: Gate = request.app.state.gate
        counts = {"admitted": gate.admitted, "calls": gate.calls, "rejected": gate.rejected}
        return {"status": "ok"} | counts

    return app


def app_from_env() -> FastAPI:
    settings = Settings()
    return create_app(settings, local_embedder(settings), local_reranker(settings))
