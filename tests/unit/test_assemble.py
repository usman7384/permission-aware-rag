import pytest

from rag.api.generate import assemble, both_ends
from rag.api.hydrate import ParentHit
from rag.common.config import Settings
from rag.common.text import tokens

SETTINGS = Settings(abstain_threshold=1, context_token_budget=6000)


def hit(name: str, score: float, words: int = 5) -> ParentHit:
    return ParentHit(
        parent_id=f"p-{name}",
        doc_id=f"d-{name}",
        version=1,
        path=f"{name}.md",
        heading=None,
        text=" ".join([name] * words),
        score=score,
    )


def names(parents: list[ParentHit]) -> list[str]:
    return [p.parent_id.removeprefix("p-") for p in parents]


def count(text: str) -> int:
    return len(tokens(text))


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        ("a", "a"),
        ("ab", "ab"),
        ("abc", "acb"),
        ("abcde", "acedb"),
    ],
)
def test_best_at_both_ends(ranked: str, expected: str) -> None:
    parents = [hit(name, score=10 - i) for i, name in enumerate(ranked)]
    assert names(both_ends(parents)) == list(expected)
    shuffled = parents[::-1]
    context = assemble(shuffled, True, SETTINGS, count, "question")
    assert context is not None
    assert names(context.parents) == list(expected)


def test_lowest_scored_parents_are_dropped_to_fit_the_budget() -> None:
    parents = [hit("a", 5, words=40), hit("b", 4, words=40), hit("c", 3, words=40)]
    settings = SETTINGS.model_copy(update={"context_token_budget": 85})
    context = assemble(parents[::-1], True, settings, count, "question")
    assert context is not None
    assert names(context.parents) == ["a", "b"]
    assert '<doc n="1" path="a.md">' in context.user
    assert '<doc n="2" path="b.md">' in context.user
    assert "c.md" not in context.user
    assert context.user.endswith("Question: question")


def test_abstains_without_parents_or_below_the_threshold() -> None:
    assert assemble([], True, SETTINGS, count, "q") is None
    assert assemble([hit("a", 0.5)], True, SETTINGS, count, "q") is None
    assert assemble([hit("a", 0.5)], False, SETTINGS, count, "q") is not None
