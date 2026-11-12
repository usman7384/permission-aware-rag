import pytest

from rag.api.retrieve import Candidate, rrf


def chunks(*ids: str) -> list[Candidate]:
    return [Candidate(chunk_id=i, parent_id=f"p-{i}", doc_id=f"d-{i}", version=1) for i in ids]


def test_scores_match_the_formula_on_a_hand_worked_example() -> None:
    fused = rrf([chunks("a", "b", "c"), chunks("c", "a", "d")], k=60, keep=10)
    assert [c.chunk_id for c in fused] == ["a", "c", "b", "d"]
    expected = [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    assert [c.score for c in fused] == pytest.approx(expected)


def test_ties_break_by_chunk_id() -> None:
    fused = rrf([chunks("y", "w"), chunks("x", "v")], k=60, keep=10)
    assert [c.chunk_id for c in fused] == ["x", "y", "v", "w"]


def test_chunk_in_both_lists_outranks_one_in_a_single_list_at_the_same_rank() -> None:
    fused = rrf([chunks("a-single", "z-both"), chunks("z-both")], k=60, keep=10)
    assert [c.chunk_id for c in fused] == ["z-both", "a-single"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_keeps_only_the_best() -> None:
    fused = rrf([chunks("a", "b", "c")], k=60, keep=2)
    assert [c.chunk_id for c in fused] == ["a", "b"]


def test_no_branches_gives_nothing() -> None:
    assert rrf([], k=60, keep=10) == []
