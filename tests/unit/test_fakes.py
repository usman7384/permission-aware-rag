import math

from rag.common.models import FakeEmbedder, FakeLLM, FakeReranker


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_embedder_is_deterministic_and_normalized() -> None:
    first = FakeEmbedder(384).embed_query("The public API allows 120 requests per minute")
    second = FakeEmbedder(384).embed_query("The public API allows 120 requests per minute")
    assert first == second
    assert len(first) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


def test_fake_embedder_gives_higher_similarity_to_shared_words() -> None:
    embedder = FakeEmbedder(384)
    query = embedder.embed_query("how many requests per minute does the API allow")
    related, unrelated = embedder.embed_documents(
        ["The public API allows 120 requests per minute per key.", "Office closes in December."]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_fake_reranker_counts_distinct_query_tokens() -> None:
    scores = FakeReranker().score("ZEBRA-7 zebra limits", ["zebra 7 zebra", "limits", "nothing"])
    assert scores == [2.0, 1.0, 0.0]


async def test_fake_llm_answers_by_prompt_kind_and_streams_words() -> None:
    llm = FakeLLM()
    answer = await llm.complete("KIND: answer\nrules", "context", model="m", max_tokens=10)
    assert answer == "Based on the documents, the answer is in [1]."
    streamed = [w async for w in llm.stream("KIND: answer", "x", model="m", max_tokens=10)]
    assert "".join(streamed) == answer
    assert len(streamed) == len(answer.split(" "))
    assert await llm.complete("KIND: judge", "x", model="m", max_tokens=1) == "1"
    assert await llm.complete("KIND: context", "x", model="m", max_tokens=1) == (
        "This chunk is from the document."
    )


async def test_fake_llm_rewrite_joins_previous_and_current_question() -> None:
    user = "user: what is ZEBRA-7\nassistant: a code word [1]\nquestion: and its limit?"
    rewritten = await FakeLLM().complete("KIND: rewrite", user, model="m", max_tokens=50)
    assert rewritten == "what is ZEBRA-7 and its limit?"
