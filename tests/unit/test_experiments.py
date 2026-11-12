import asyncio
import json
import random
import time
from collections import Counter, defaultdict

import pytest

from rag.api.pipeline import Outcome
from rag.common.config import Settings
from rag.common.models import FakeLLM
from rag.common.text import tokens
from rag.evals.benchmark import BenchDoc
from rag.evals.experiments.ablation import variant_ms
from rag.evals.experiments.cache_behaviour import Logged, negated, paraphrases, query_log, replay
from rag.evals.experiments.common import (
    CountingLLM,
    overlap,
    sample_corpus,
    table,
    variant_settings,
)
from rag.evals.experiments.cost import Prices, Tokens, usd_per_1k
from rag.evals.experiments.failure_injection import Trial, answer_kind, summarize
from rag.evals.experiments.filtered_search import readable_groups, with_folder_principals
from rag.evals.experiments.freshness import scratch_redis, seconds_until
from rag.evals.experiments.loadtest import locust_summary, stage_percentiles
from rag.evals.experiments.scale_memory import ram_mb
from rag.evals.score import Golden
from rag.evals.synthetic import synthetic_acl


def test_table_aligns_columns_and_rounds_floats() -> None:
    text = table(["name", "value"], [["a", 0.12345], ["long name", 2]])
    assert text.splitlines() == ["     name  value", "        a  0.123", "long name      2"]


def test_sample_keeps_relevant_documents_of_the_first_queries_then_fills_in_corpus_order() -> None:
    docs = [BenchDoc(str(i), "", "") for i in range(1, 9)]
    qrels = {"q0": {"9": 0}, "q1": {"7": 1, "2": 0}, "q2": {"5": 2}, "q3": {"8": 1}}
    assert [d.id for d in sample_corpus(docs, qrels, 2, 4)] == ["1", "2", "5", "7"]
    assert [d.id for d in sample_corpus(docs, qrels, 2, 1)] == ["5", "7"]
    assert [d.id for d in sample_corpus(docs, qrels, None, 3)] == ["5", "7", "8"]
    missing = qrels | {"q4": {"99": 1}}
    assert [d.id for d in sample_corpus(docs, missing, None, 4)] == ["1", "5", "7", "8"]


def test_variant_settings_get_their_own_database_and_index_version() -> None:
    base = Settings(pg_dsn="postgresql://rag:rag@127.0.0.1:5432/rag_bench", store_prefix="bench_")
    variant = variant_settings(base, "c64p", child_tokens=64)
    assert "dbname=rag_bench_c64p" in variant.pg_dsn
    assert variant.chunks_name == "bench_chunks_c64p" and variant.child_tokens == 64
    assert base.chunks_name == "bench_chunks_v1"


def test_overlap_is_the_share_of_true_neighbours_found() -> None:
    assert overlap(["a", "b", "x"], ["a", "b", "c", "d"]) == 0.5
    assert overlap([], []) == 1.0


def test_ram_follows_bytes_per_dimension() -> None:
    assert ram_mb(1024, 256, "none") == 1.0
    assert ram_mb(1024, 256, "int8") == 0.25
    assert ram_mb(1024, 256, "binary") == 1 / 32


def test_hybrid_waits_for_the_slower_branch_and_rerank_adds_on() -> None:
    ms = variant_ms({"embed": 5.0, "dense": 10.0, "sparse": 12.0, "rerank": 100.0})
    assert ms == {"sparse": 12.0, "dense": 15.0, "hybrid": 15.0, "hybrid+rerank": 115.0}


def test_points_get_their_folder_group_and_the_user_reads_the_first_folders() -> None:
    points = [({"doc_id": d, "principals": ["old"]}, [0.0]) for d in ("b", "a", "c", "a")]
    stamped = with_folder_principals(points, 2, "user:bench")
    assert [p["principals"] for p, _ in stamped] == [
        ["group:filtered-f0001"],
        ["group:filtered-f0000"],
        ["group:filtered-f0000"],
        ["group:filtered-f0000"],
    ]
    acl = synthetic_acl("filtered", 3, 2, 0.5, "user:bench")
    assert readable_groups(acl, "user:bench") == ["group:filtered-f0000"]


async def test_counting_llm_counts_calls_and_tokens() -> None:
    llm = CountingLLM(FakeLLM(), lambda text: len(tokens(text)))
    reply = await llm.complete("KIND: context", "one two", model="", max_tokens=8)
    assert reply == "This chunk is from the document."
    assert (llm.calls, llm.prompt_tokens, llm.completion_tokens) == (1, 4, 6)


def test_paraphrases_keep_the_meaning_and_the_negation_flips_it() -> None:
    assert paraphrases("Alpha beta gamma.") == [
        "Alpha beta gamma.",
        "alpha beta gamma",
        "Is it true that alpha beta gamma?",
        "Tell me whether alpha beta gamma.",
    ]
    assert negated("Alpha beta gamma.") == "It is false that alpha beta gamma."


def test_query_log_is_skewed_towards_the_first_intents_and_repeatable() -> None:
    texts = [f"Claim {i}." for i in range(20)]
    log = query_log(texts, 2000, 1.0, 0.1, random.Random(1))
    assert log == query_log(texts, 2000, 1.0, 0.1, random.Random(1))
    counts = Counter(q.intent for q in log)
    assert counts[0] > counts[1] > counts[19] and counts[0] > counts[20]
    assert {q.text for q in log if q.intent == 20} == {"It is false that claim 0."}


def test_replay_counts_hits_and_a_similar_negation_is_a_miss() -> None:
    first, asked, negation = "Alpha beta", "Is it true that alpha beta?", "It is false that alpha"
    other = "Alpha beta delta"
    log = [Logged(first, 0), Logged("alpha beta.", 0), Logged(asked, 0), Logged(negation, 1)]
    log.append(Logged(other, 2))
    similarity = defaultdict(
        lambda: 0.5, {(first, asked): 0.97, (first, negation): 0.98, (first, other): 0.985}
    )
    assert replay(log, similarity, 0.95) == pytest.approx(
        {
            "hit_rate": 0.6,
            "exact": 0.2,
            "semantic": 0.4,
            "false_hits": 0.2,
            "missed_reuse": 0.0,
        }
    )
    strict = replay(log, similarity, 0.99)
    assert strict["hit_rate"] == strict["exact"] == strict["missed_reuse"] == 0.2
    assert strict["false_hits"] == 0.0


def test_cost_by_stage_with_prompt_and_answer_caches() -> None:
    prices = Prices(3.0, 0.3, 15.0, 1.0, 5.0, min_cached_tokens=1024)
    tokens = Tokens(100, 20, answer_in=2000, answer_system=1500, answer_out=100)
    assert usd_per_1k(tokens, prices, 0.5, 0.0, False) == {
        "rewrite": 0.1,
        "answer": 7.5,
        "total": 7.6,
    }
    assert usd_per_1k(tokens, prices, 0.5, 0.0, True)["answer"] == pytest.approx(3.45)
    assert usd_per_1k(tokens, prices, 0.5, 0.5, False)["answer"] == 3.75
    short = Tokens(100, 20, 2000, 150, 100)
    assert usd_per_1k(short, prices, 0.5, 0.0, True) == usd_per_1k(short, prices, 0.5, 0.0, False)


def test_answer_kind_and_fault_summary() -> None:
    assert answer_kind(Outcome(abstained=True)) == "abstained"
    assert answer_kind(Outcome(degraded=["llm"])) == "extractive"
    assert answer_kind(Outcome(degraded=["rerank"])) == "llm"
    questions = [Golden("1", "u", "q", ["a"]), Golden("2", "u", "q", ["c"])]
    trials = [Trial(["a", "b"], [], "llm", 10.0), Trial([], ["dense", "sparse"], "error", 30.0)]
    summary = summarize(questions, trials)
    assert summary["recall@10"] == 0.5 and summary["ndcg@10"] == 0.5
    assert (summary["p50_ms"], summary["p95_ms"]) == (10.0, 30.0)
    assert (summary["llm"], summary["error"], summary["extractive"]) == (0.5, 0.5, 0.0)
    assert summary["degraded"] == "dense,sparse"


async def test_seconds_until_polls_until_the_check_passes() -> None:
    calls = []

    async def third_time() -> bool:
        calls.append(1)
        return len(calls) == 3

    assert await seconds_until(third_time, time.perf_counter(), 1.0, 0.001) < 1.0
    assert len(calls) == 3
    with pytest.raises(TimeoutError):
        await seconds_until(lambda: asyncio.sleep(0, False), time.perf_counter(), 0.01, 0.001)


def test_scratch_redis_changes_the_database_and_refuses_the_configured_one() -> None:
    assert scratch_redis("redis://127.0.0.1:6380/0", 1) == "redis://127.0.0.1:6380/1"
    with pytest.raises(ValueError, match="scratch"):
        scratch_redis("redis://127.0.0.1:6380/0", 0)
    with pytest.raises(ValueError, match="scratch"):
        scratch_redis("redis://127.0.0.1:6380", 0)


def test_stage_percentiles_skip_requests_without_the_stage() -> None:
    full = {"embed": 1.0, "retrieve": 2.0, "rrf": 0.0, "recheck": 1.0, "rerank": 10.0}
    full |= {"hydrate": 1.0, "generate": 5.0, "output_guard": 1.0, "persist": 2.0}
    hit = {"exact_cache": 1.0, "persist": 4.0}
    stages = stage_percentiles([full, hit])
    assert stages["retrieval"] == {"n": 1, "p50": 15.0, "p95": 15.0, "p99": 15.0}
    assert stages["generation"]["p50"] == 6.0 and stages["persist"]["n"] == 2


def test_locust_summary_reads_the_named_entry() -> None:
    entry = {"num_requests": 4, "num_failures": 1, "start_time": 100.0}
    ask = entry | {"name": "ask", "last_request_timestamp": 102.0}
    ask["response_times"] = {"350": 2, "900": 1, "1200": 1}
    hit = entry | {"name": "hit", "last_request_timestamp": 100.0, "response_times": {"5": 4}}
    stats = json.dumps([ask, hit])
    assert locust_summary(stats) == {
        "n": 4,
        "failures": 1,
        "p50": 350.0,
        "p95": 1200.0,
        "p99": 1200.0,
        "rps": 2.0,
    }
    assert locust_summary(stats, "hit")["p95"] == 5.0 and locust_summary(stats, "hit")["rps"] == 0
