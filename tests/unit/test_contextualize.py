from rag.common.config import Settings
from rag.common.models import FakeEmbedder, FakeLLM
from rag.worker.chunk import Parent
from rag.worker.contextualize import contextualize, first_two_sentences, head

COUNT = FakeEmbedder(8).count_tokens


def test_head_keeps_the_longest_prefix_within_the_limit() -> None:
    assert head("one two three four", 2, COUNT) == "one two "
    assert head("one two", 5, COUNT) == "one two"
    assert head("", 3, COUNT) == ""


def test_reply_is_cut_to_two_sentences_on_one_line() -> None:
    assert first_two_sentences(" First one.\nSecond  one! Third one. ") == "First one. Second one!"


async def test_one_context_per_parent_from_the_context_prompt() -> None:
    parents = [Parent(None, "a.", ["a."]), Parent("B", "# B\nb.", ["# B\n", "b."])]
    contexts = await contextualize(FakeLLM(), Settings(), "a. # B b.", parents, COUNT)
    assert contexts == ["This chunk is from the document."] * 2
