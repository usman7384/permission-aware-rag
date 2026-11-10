from pathlib import Path

from rag.common import redis_
from rag.common.keys import DEAD_STREAM, JOBS_GROUP, JOBS_STREAM
from rag.worker.index import Deps
from rag.worker.main import consume_once, reclaim_once


async def test_failing_job_is_dead_lettered_after_max_deliveries(deps: Deps) -> None:
    deps.settings = deps.settings.model_copy(update={"reclaim_idle_ms": 0})
    # Invalid UTF-8 makes the parser raise on every attempt.
    (Path(deps.settings.corpus_dir) / "broken.md").write_bytes(b"# Broken\n\xff\xfe\n")
    await redis_.ensure_jobs_group(deps.redis)
    job_id = await redis_.enqueue_job(deps.redis, "upsert", "broken.md")

    assert await consume_once(deps, "test") == 1
    for _ in range(deps.settings.max_deliveries - 2):
        assert await reclaim_once(deps, "test") == 1
    assert await deps.redis.xlen(DEAD_STREAM) == 0

    assert await reclaim_once(deps, "test") == 1
    [(_, dead)] = await deps.redis.xrange(DEAD_STREAM)
    assert dead["job_id"] == job_id
    assert dead["deliveries"] == str(deps.settings.max_deliveries)
    assert "UnicodeDecodeError" in dead["error"]
    assert (await deps.redis.xpending(JOBS_STREAM, JOBS_GROUP))["pending"] == 0
    assert await reclaim_once(deps, "test") == 0
