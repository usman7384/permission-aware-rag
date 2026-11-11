import argparse
import asyncio
import csv
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import uvicorn

from rag.api import pipeline
from rag.api.main import app
from rag.common import pg
from rag.common.config import Settings
from rag.common.keys import api_key_hash
from rag.evals.experiments import common
from rag.evals.score import percentile

LOCUSTFILE = Path(__file__).parents[1] / "loadtest.py"
API_KEY = "bench-loadtest"
# Retrieval is steps 6 and 8 to 12, generation steps 14 and 15.
STAGES = {
    "retrieval": ("embed", "retrieve", "rrf", "recheck", "rerank", "hydrate"),
    "rerank": ("rerank",),
    "generation": ("generate", "output_guard"),
    "persist": ("persist",),
}


def stage_percentiles(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """p50, p95 and p99 of each stage over the requests that ran all of its steps."""
    result = {}
    for stage, steps in STAGES.items():
        values = [sum(t[s] for s in steps) for t in timings if all(s in t for s in steps)]
        if values:
            result[stage] = {"n": len(values)} | {
                f"p{p}": percentile(values, p) for p in (50, 95, 99)
            }
    return result


def locust_summary(stats_csv: str) -> dict[str, float]:
    """Requests, failures, end-to-end percentiles and throughput of the `ask` row of Locust's
    `_stats.csv`."""
    row = next(r for r in csv.DictReader(stats_csv.splitlines()) if r["Name"] == "ask")
    return {
        "n": int(row["Request Count"]),
        "failures": int(row["Failure Count"]),
        **{f"p{p}": float(row[f"{p}%"]) for p in (50, 95, 99)},
        "rps": float(row["Requests/s"]),
    }


def run_locust(args: argparse.Namespace, users: int, questions: Path, prefix: str) -> str:
    """Run Locust headless at `users` concurrent users; returns its `_stats.csv`. Locust runs
    in its own process because it monkey-patches the standard library with gevent."""
    out = questions.parent / f"u{users}"
    command = [sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless"]
    command += ["--host", f"http://127.0.0.1:{args.port}", "--users", str(users)]
    command += ["--spawn-rate", str(users), "--run-time", f"{args.seconds}s"]
    command += ["--stop-timeout", f"{args.stop_timeout_s}s", "--exit-code-on-error", "0"]
    command += ["--csv", str(out), "--only-summary", "--questions", str(questions)]
    command += ["--api-key", API_KEY, "--session-prefix", prefix]
    timeout = args.seconds + args.stop_timeout_s + 60
    subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    return Path(f"{out}_stats.csv").read_text(encoding="utf-8")


async def level(
    deps: pipeline.Deps, args: argparse.Namespace, users: int, questions: list[str], folder: Path
) -> dict[str, Any]:
    """One concurrency level: Locust's end-to-end numbers and the server's per-stage timings."""
    file = folder / f"questions-u{users}.json"
    file.write_text(json.dumps(questions), encoding="utf-8")
    prefix = f"load-{uuid.uuid4().hex[:8]}-u{users}-"
    stats = await asyncio.to_thread(run_locust, args, users, file, prefix)
    async with deps.pg.connection() as conn:
        rows = await pg.select_request_timings(conn, prefix)
    hits = sum(1 for tier, _ in rows if tier != "none")
    stages = stage_percentiles([t for tier, t in rows if tier == "none"])
    return {"request": locust_summary(stats), **stages, "cache_hits": hits}


async def serve(deps: pipeline.Deps, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the API in this process on the benchmark stores while Locust loads it."""
    app.state.deps = deps
    config = uvicorn.Config(
        app, host="127.0.0.1", port=args.port, lifespan="off", log_level="warning"
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.05)
        texts = [q.question for q in await common.labelled_questions(deps.pg, args)]
        # Each level asks its own questions, so earlier levels do not warm the answer caches.
        per_level = len(texts) // len(args.users)
        if per_level == 0:
            raise ValueError("fewer labelled questions than concurrency levels")
        with tempfile.TemporaryDirectory() as folder:
            slices = [texts[i * per_level : (i + 1) * per_level] for i in range(len(args.users))]
            return [
                await level(deps, args, users, questions, Path(folder))
                for users, questions in zip(args.users, slices, strict=True)
            ]
    finally:
        server.should_exit = True
        await serving


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    # The load comes from one API key, so the per-key rate limit is lifted for this run only.
    settings = settings.model_copy(update={"rate_limit_per_min": 10**9})
    deps = await pipeline.open_deps(settings)
    deps.llm = common.fake_llm_unless_configured(settings) or deps.llm
    try:
        async with deps.pg.connection() as conn:
            await pg.upsert_api_key(conn, api_key_hash(API_KEY), args.user, False)
        levels = await serve(deps, args)
    finally:
        await pipeline.close_deps(deps)
    llm_used = common.llm_name(settings)
    runs = dict(zip(map(str, args.users), levels, strict=True))
    await common.save_run(settings, args, {"llm": llm_used, "users": runs})
    headers = ["users", "stage", "n", "p50_ms", "p95_ms", "p99_ms", "rps", "failures"]
    rows = []
    for users, result in zip(args.users, levels, strict=True):
        r = result["request"]
        rows.append([users, "request", r["n"], r["p50"], r["p95"], r["p99"], r["rps"]])
        rows[-1].append(r["failures"])
        for stage in STAGES:
            if stage in result:
                s = result[stage]
                rows.append([users, stage, s["n"], s["p50"], s["p95"], s["p99"], "", ""])
    footer = f"\n{args.seconds} s per level, {llm_used} LLM; request = end to end, from Locust"
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    description = "Per-stage latency and throughput of POST /ask at rising concurrency."
    parser = common.parser(description, "loadtest")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8], help="levels")
    parser.add_argument("--seconds", type=int, default=15, help="duration of each level")
    parser.add_argument("--stop-timeout-s", type=int, default=30, help="let requests finish")
    parser.add_argument("--port", type=int, default=8765, help="port of the API under load")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
