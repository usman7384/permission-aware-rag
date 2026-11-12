import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from test_ask import ask
from test_cache import QUESTION, done

from rag.api.pipeline import Deps
from rag.common.config import Settings
from rag.common.models import (
    FakeEmbedder,
    FakeReranker,
    RemoteReranker,
    inference_client,
    make_embedder,
    make_reranker,
)
from rag.inference.main import create_app

TEXTS = ["Access tokens expire after 15 minutes.", "The office closes on 24 December."]


@contextmanager
def serving(app: FastAPI, port: int) -> Iterator[str]:
    """Serve the app with uvicorn in a background thread; yields its URL."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join()


class BlockingReranker:
    """Holds every call until `release` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.release.wait(30)
        return [0.0] * len(passages)


@pytest.fixture
def fake_service(settings: Settings) -> Iterator[str]:
    app = create_app(settings, FakeEmbedder(settings.embed_dim), FakeReranker())
    with serving(app, 8791) as url:
        yield url


@pytest.fixture
def full_service(settings: Settings) -> Iterator[tuple[FastAPI, str]]:
    """A service with one model slot and no queue, its slot held by a blocked rerank call."""
    one_slot = settings.model_copy(update={"inference_concurrency": 1, "inference_queue_max": 0})
    blocking = BlockingReranker()
    app = create_app(one_slot, FakeEmbedder(settings.embed_dim), blocking)
    with serving(app, 8792) as url:
        holder = RemoteReranker(inference_client(url, 30000))
        thread = threading.Thread(target=holder.score, args=("q", ["p"]))
        thread.start()
        while app.state.gate.admitted < 1:
            time.sleep(0.01)
        try:
            yield app, url
        finally:
            blocking.release.set()
            thread.join()


async def test_remote_clients_match_the_in_process_fakes(
    settings: Settings, fake_service: str
) -> None:
    remote = settings.model_copy(update={"inference_url": fake_service})
    embedder, fake = make_embedder(remote), FakeEmbedder(settings.embed_dim)
    assert embedder.embed_documents(TEXTS) == fake.embed_documents(TEXTS)
    assert embedder.embed_query(TEXTS[0]) == fake.embed_query(TEXTS[0])
    assert embedder.count_tokens(TEXTS[0]) == fake.count_tokens(TEXTS[0])
    assert make_reranker(remote).score("access tokens", TEXTS) == FakeReranker().score(
        "access tokens", TEXTS
    )


async def test_full_queue_returns_503_at_once_and_the_api_degrades_rerank(
    client: httpx.AsyncClient,
    api_deps: Deps,
    full_service: tuple[FastAPI, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, url = full_service
    start = time.perf_counter()
    async with httpx.AsyncClient(base_url=url) as service:
        response = await service.post("/rerank", json={"query": "q", "passages": ["p"]})
    assert response.status_code == 503 and time.perf_counter() - start < 1
    monkeypatch.setattr(api_deps, "embedder", FakeEmbedder(api_deps.settings.embed_dim))
    monkeypatch.setattr(api_deps, "reranker", RemoteReranker(inference_client(url, 30000)))
    events = await ask(client, "dev-ana", QUESTION)
    assert done(events)["degraded"] == ["rerank"]
    assert app.state.gate.rejected == 2


async def test_embed_refused_on_the_query_path_is_an_error_event(
    client: httpx.AsyncClient,
    api_deps: Deps,
    full_service: tuple[FastAPI, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, url = full_service
    remote = api_deps.settings.model_copy(update={"inference_url": url})
    monkeypatch.setattr(api_deps, "embedder", make_embedder(remote))
    events = await ask(client, "dev-ana", QUESTION)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["code"] == "embedding_unavailable"


async def test_healthz_reports_inference(
    client: httpx.AsyncClient, api_deps: Deps, fake_service: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_deps.settings, "inference_url", fake_service)
    response = await client.get("/healthz")
    assert response.status_code == 200 and response.json()["checks"]["inference"] is True
    monkeypatch.setattr(api_deps.settings, "inference_url", "http://127.0.0.1:1")
    response = await client.get("/healthz")
    assert response.status_code == 503 and response.json()["checks"]["inference"] is False
