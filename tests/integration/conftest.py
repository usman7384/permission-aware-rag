import os
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
from fixture_corpus import CORPUS
from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo

from rag.api import pipeline
from rag.api.auth import authenticate
from rag.api.main import app
from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.config import Settings
from rag.common.keys import JOBS_GROUP, JOBS_STREAM
from rag.common.migrate import migrate
from rag.common.seed import seed
from rag.worker.index import Deps
from rag.worker.main import close_deps, consume_once, open_deps
from rag.worker.watch import reconcile

TEST_DB = "test_rag"
DEV_SETTINGS = Settings()
TEST_SETTINGS = Settings(
    pg_dsn=make_conninfo(DEV_SETTINGS.pg_dsn, dbname=TEST_DB),
    redis_url=urlunsplit(urlsplit(DEV_SETTINGS.redis_url)._replace(path="/15")),
    store_prefix="test_",
    model_backend="fake",
    abstain_threshold=1,
    worker_block_ms=100,
    # Store setup can take over 5 s on a busy laptop.
    store_timeout_s=30,
)
# The API reads Settings() from the environment.
os.environ.update(
    PG_DSN=TEST_SETTINGS.pg_dsn,
    REDIS_URL=TEST_SETTINGS.redis_url,
    STORE_PREFIX=TEST_SETTINGS.store_prefix,
    MODEL_BACKEND=TEST_SETTINGS.model_backend,
    ABSTAIN_THRESHOLD=str(TEST_SETTINGS.abstain_threshold),
)


def wait_until_serving(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"{url} did not start serving in {timeout_s} s")


@pytest.fixture(scope="session", autouse=True)
def inference_service() -> Iterator[None]:
    url = TEST_SETTINGS.inference_url
    if not url:
        yield
        return
    parts = urlsplit(url)
    env = {k: v for k, v in os.environ.items() if k != "INFERENCE_URL"} | {"MODEL_BACKEND": "fake"}
    command = [sys.executable, "-m", "uvicorn", "--factory", "rag.inference.main:app_from_env"]
    command += ["--host", parts.hostname or "", "--port", str(parts.port)]
    command += ["--log-level", "warning"]
    server = subprocess.Popen(command, env=env)
    try:
        wait_until_serving(url, timeout_s=60)
        yield
    finally:
        server.terminate()
        server.wait(timeout=30)


@pytest.fixture(scope="session")
async def settings() -> Settings:
    await pg.ensure_database(DEV_SETTINGS.pg_dsn, TEST_DB, DEV_SETTINGS.store_timeout_s)
    await migrate(TEST_SETTINGS, job_id="test-migrate")
    return TEST_SETTINGS


async def _reset_stores(settings: Settings) -> None:
    assert settings.store_prefix == "test_" and settings.redis_url.endswith("/15")
    async with await AsyncConnection.connect(settings.pg_dsn) as conn:
        await conn.execute(
            "TRUNCATE documents, tuples, api_keys, turns, requests, eval_runs, outbox CASCADE"
        )
        await seed(conn)
    qdrant = qdrant_.connect(settings)
    opensearch = opensearch_.connect(settings)
    try:
        await qdrant.delete_collection(settings.chunks_name)
        await qdrant.delete_collection(settings.answer_cache_name)
        await opensearch.indices.delete(index=settings.chunks_name, ignore_unavailable=True)
    finally:
        await qdrant.close()
        await opensearch.close()
    await migrate(settings, job_id="test-reset")


@pytest.fixture
async def deps(settings: Settings, tmp_path: Path) -> AsyncIterator[Deps]:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    worker_settings = settings.model_copy(update={"corpus_dir": str(corpus)})
    await _reset_stores(worker_settings)
    deps = await open_deps(worker_settings)
    await deps.redis.flushdb()
    try:
        yield deps
    finally:
        await close_deps(deps)


@pytest.fixture
async def ingested(deps: Deps) -> Deps:
    await redis_.ensure_group(deps.redis, JOBS_STREAM, JOBS_GROUP)
    await reconcile(deps.settings, deps.pg, deps.redis)
    while await consume_once(deps, "test"):
        pass
    return deps


@pytest.fixture
async def client(ingested: Deps) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def api_deps(client: httpx.AsyncClient) -> pipeline.Deps:
    return app.state.deps


async def search_paths(deps: pipeline.Deps, api_key: str, question: str) -> set[str]:
    async with deps.pg.connection() as conn:
        caller = await authenticate(conn, deps.redis, deps.settings, api_key)
        principals, _ = await pipeline.expand(conn, caller.user_id)
    vector = await pipeline.embed(deps.embedder, question)
    parents, degraded = await pipeline.search(deps, principals, question, vector, "test", {})
    assert degraded == []
    return {parent.path for parent in parents}
