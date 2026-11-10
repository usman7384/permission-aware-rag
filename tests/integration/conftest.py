import os
from urllib.parse import urlsplit, urlunsplit

import pytest
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo

from rag.common.config import Settings
from rag.common.migrate import migrate

TEST_DB = "test_rag"
DEV_SETTINGS = Settings()
TEST_SETTINGS = Settings(
    pg_dsn=make_conninfo(DEV_SETTINGS.pg_dsn, dbname=TEST_DB),
    redis_url=urlunsplit(urlsplit(DEV_SETTINGS.redis_url)._replace(path="/15")),
    store_prefix="test_",
    model_backend="fake",
    abstain_threshold=1,
)
# The API reads Settings() from the environment, so point it at the test stores.
os.environ.update(
    PG_DSN=TEST_SETTINGS.pg_dsn,
    REDIS_URL=TEST_SETTINGS.redis_url,
    STORE_PREFIX=TEST_SETTINGS.store_prefix,
    MODEL_BACKEND=TEST_SETTINGS.model_backend,
    ABSTAIN_THRESHOLD=str(TEST_SETTINGS.abstain_threshold),
)


async def _create_test_database() -> None:
    async with await AsyncConnection.connect(DEV_SETTINGS.pg_dsn, autocommit=True) as conn:
        cur = await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
        if await cur.fetchone() is None:
            await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))


@pytest.fixture(scope="session")
async def settings() -> Settings:
    await _create_test_database()
    await migrate(TEST_SETTINGS, job_id="test-migrate")
    return TEST_SETTINGS
