import math

import pytest

from rag.evals.score import (
    latency,
    ndcg_at,
    percentile,
    recall_at,
    reciprocal_rank,
)


def test_recall_counts_expected_paths_in_the_top_k() -> None:
    assert recall_at(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at(["a", "b"], ["a", "z"], 10) == 0.5


def test_reciprocal_rank_uses_the_first_expected_path() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_percentiles_use_the_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([7.0], 95) == 7.0
    assert latency([{"a": 1.0}, {"a": 3.0, "b": 2.0}]) == {
        "a": {"p50": 1.0, "p95": 3.0},
        "b": {"p50": 2.0, "p95": 2.0},
    }


def test_ndcg_matches_a_hand_worked_example() -> None:
    relevance = {"a": 2, "b": 1}
    assert ndcg_at(["a", "b", "c"], relevance, 10) == 1.0
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at(["b", "a"], relevance, 10) == pytest.approx(expected)
    assert ndcg_at(["c", "a"], {"a": 1}, 1) == 0.0
    assert ndcg_at(["c"], {}, 10) == 0.0
