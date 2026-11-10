from collections.abc import AsyncIterator

from rag.api.generate import closing_events, output_guard
from rag.api.hydrate import ParentHit

PARENTS = [
    ParentHit("p1", "d1", 1, "a.md", None, "Write to ops@example.com for access.", 2.0),
    ParentHit("p2", "d2", 1, "b.md", None, "Tokens expire after 15 minutes.", 1.0),
]


async def pieces(text: str) -> AsyncIterator[str]:
    for word in text.split(" "):
        yield word + " "


async def guarded(text: str) -> list[str]:
    return [sentence async for sentence in output_guard(pieces(text), PARENTS)]


async def test_out_of_range_citation_marker_is_removed() -> None:
    released = await guarded("Tokens expire after 15 minutes [2][3]. See [0] too [1].")
    answer = "".join(released)
    assert "[3]" not in answer and "[0]" not in answer
    assert "[2]" in answer and "[1]" in answer
    [(name, data)] = closing_events(answer, PARENTS)
    assert name == "citations"
    assert [item["n"] for item in data["items"]] == [1, 2]
    assert data["items"][0] == {"n": 1, "doc_id": "d1", "path": "a.md", "parent_id": "p1"}


async def test_released_one_sentence_at_a_time() -> None:
    released = await guarded("First one [1]. Second one [2]. Third")
    assert released == ["First one [1]. ", "Second one [2]. ", "Third "]


async def test_email_not_in_context_is_redacted_and_one_in_context_is_kept() -> None:
    answer = "".join(await guarded("Ask ops@example.com or eve@evil.test [1]."))
    assert "ops@example.com" in answer
    assert "eve@evil.test" not in answer
    assert "[redacted]" in answer


async def test_long_number_not_in_context_is_redacted() -> None:
    answer = "".join(await guarded("Card 4111111111111111 works [1]."))
    assert "4111111111111111" not in answer


async def test_answer_without_valid_citation_yields_the_uncited_warning() -> None:
    answer = "".join(await guarded("Nothing to cite here [7]."))
    assert closing_events(answer, PARENTS) == [
        ("citations", {"items": []}),
        ("warning", {"code": "uncited"}),
    ]
