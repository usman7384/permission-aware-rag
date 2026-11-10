import pytest

from rag.common.text import extract_entities, normalize_query


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What is ZEBRA-7?", "what is zebra-7"),
        ("  How   many\tdays\nof leave?  ", "how many days of leave"),
        ("Deploy days!!!", "deploy days"),
        ("Is it open at 8:00...", "is it open at 8:00"),
        ("ＡＰＩ limits?", "api limits"),
        ("what is v1.5", "what is v1.5"),
        ("Who approves? ", "who approves"),
        ("Why?!", "why"),
    ],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is ZEBRA-7?", ["7", "zebra"]),
        ("how many days of leave", []),
        ("Expenses above 500 euros?", ["500"]),
        ("Where is the atlas-blobs bucket?", []),
        ("Does OpenSearch use BM25?", ["bm25", "opensearch"]),
        ("What does RFC mean, and RFC again?", ["rfc"]),
        ("is the office open 8:00 to 19:00", ["00", "19", "8"]),
        ("Compare iPhone and v2", ["iphone", "v2"]),
    ],
)
def test_extract_entities(question: str, expected: list[str]) -> None:
    assert extract_entities(question) == expected
