import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
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


def locust_summary(stats_json: str, name: str = "ask") -> dict[str, float]:
    """Requests, failures, end-to-end percentiles and throughput of the entry `name` of Locust's
    final `--json-file` stats, whose `response_times` maps rounded milliseconds to counts."""
    entry = next(e for e in json.loads(stats_json) if e["name"] == name)
    times = [float(ms) for ms, n in entry["response_times"].items() for _ in range(n)]
    duration = entry["last_request_timestamp"] - entry["start_time"]
    return {
        "n": entry["num_requests"],
        "failures": entry["num_failures"],
        **{f"p{p}": percentile(times, p) for p in (50, 95, 99)},
        "rps": entry["num_requests"] / duration if duration > 0 else 0.0,
    }


def run_locust(
    args: argparse.Namespace, users: int, questions: Path, prefix: str, hit_question: str
) -> str:
    """Run Locust headless at `users` concurrent askers, plus the prober with `--hits`; returns
    its final stats as JSON (its `_stats.csv` is only a snapshot taken every second, which a
    short run can end before any request is in it). Locust runs in its own process because it
    monkey-patches the standard library with gevent."""
    out = questions.parent / f"u{users}"
    classes = ["Asker", "Prober"] if args.hits else ["Asker"]
    total = users + len(classes) - 1
    command = [sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless", *classes]
    command += ["--host", f"http://127.0.0.1:{args.port}", "--users", str(total)]
    command += ["--spawn-rate", str(total), "--run-time", f"{args.seconds}s"]
    command += ["--hit-question", hit_question]
    command += ["--stop-timeout", f"{args.stop_timeout_s}s", "--exit-code-on-error", "0"]
    command += ["--json-file", str(out), "--questions", str(questions)]
    command += ["--api-key", API_KEY, "--session-prefix", prefix]
    timeout = args.seconds + args.stop_timeout_s + 60
    subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    return Path(f"{out}.json").read_text(encoding="utf-8")


async def level(
    deps: pipeline.Deps,
    args: argparse.Namespace,
    users: int,
    questions: list[str],
    hit_question: str,
    folder: Path,
) -> dict[str, Any]:
    """One concurrency level: Locust's end-to-end numbers and the server's per-stage timings."""
    file = folder / f"questions-u{users}.json"
    file.write_text(json.dumps(questions), encoding="utf-8")
    prefix = f"load-{uuid.uuid4().hex[:8]}-u{users}-"
    before = await inference_counts(deps.settings)
    stats = await asyncio.to_thread(run_locust, args, users, file, prefix, hit_question)
    after = await inference_counts(deps.settings)
    async with deps.pg.connection() as conn:
        rows = await pg.select_request_timings(conn, prefix)
    full = [(degraded, t) for tier, degraded, t in rows if tier == "none"]
    result = {"request": locust_summary(stats), **stage_percentiles([t for _, t in full])}
    result["cache_hits"] = len(rows) - len(full)
    result["degraded_rerank"] = sum("rerank" in d for d, _ in full) / len(full) if full else 0.0
    if args.hits:
        result["hit"] = locust_summary(stats, "hit")
    if before and after:
        calls = after["calls"] - before["calls"]
        result["inference_503"] = (after["rejected"] - before["rejected"]) / calls if calls else 0.0
    return result


async def inference_counts(settings: Settings) -> dict[str, int] | None:
    """The inference service's call and 503 counters, if `INFERENCE_URL` is set."""
    if not settings.inference_url:
        return None
    async with httpx.AsyncClient(base_url=settings.inference_url, timeout=10) as client:
        response = await client.get("/healthz")
        response.raise_for_status()
        return response.json()


async def warm_hit_question(port: int, questions: list[str], tries: int = 10) -> str:
    """The first question that is cached after one ask (one that is answered, not abstained
    or degraded), so the prober only sees exact hits."""
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
        for question in questions[:tries]:
            for _ in range(2):
                body = {"session_id": str(uuid.uuid4()), "question": question}
                response = await client.post("/ask", json=body, headers={"X-API-Key": API_KEY})
            if '"cache_tier": "exact"' in response.text:
                return question
    raise RuntimeError(f"none of the first {tries} questions was cached")


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
        # The prober repeats one cached question, which the levels then leave out.
        hit_question = await warm_hit_question(args.port, texts) if args.hits else ""
        texts = [text for text in texts if text != hit_question]
        # Each level asks its own questions, so earlier levels do not warm the answer caches.
        per_level = len(texts) // len(args.users)
        if per_level == 0:
            raise ValueError("fewer labelled questions than concurrency levels")
        with tempfile.TemporaryDirectory() as folder:
            slices = [texts[i * per_level : (i + 1) * per_level] for i in range(len(args.users))]
            return [
                await level(deps, args, users, questions, hit_question, Path(folder))
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
    headers += ["degraded_rerank", "inference_503"]
    rows = []
    for users, result in zip(args.users, levels, strict=True):
        r = result["request"]
        rows.append([users, "request", r["n"], r["p50"], r["p95"], r["p99"], r["rps"]])
        rows[-1] += [r["failures"], result["degraded_rerank"], result.get("inference_503", "")]
        if "hit" in result:
            h = result["hit"]
            rows.append([users, "hit", h["n"], h["p50"], h["p95"], h["p99"], h["rps"]])
            rows[-1] += [h["failures"], "", ""]
        for stage in STAGES:
            if stage in result:
                s = result[stage]
                rows.append([users, stage, s["n"], s["p50"], s["p95"], s["p99"], "", "", "", ""])
    footer = f"\n{args.seconds} s per level, {llm_used} LLM; request = end to end, from Locust"
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    description = "Per-stage latency and throughput of POST /ask at rising concurrency."
    parser = common.parser(description, "loadtest")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8], help="levels")
    parser.add_argument("--seconds", type=int, default=15, help="duration of each level")
    parser.add_argument("--stop-timeout-s", type=int, default=30, help="let requests finish")
    parser.add_argument("--port", type=int, default=8765, help="port of the API under load")
    hits_help = "add one user that repeats a cached question (latency without the models)"
    parser.add_argument("--hits", action="store_true", help=hits_help)
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
