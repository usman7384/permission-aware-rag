import pytest
from fixture_corpus import CODE_WORDS, CORPUS, READABLE


def test_corpus_has_exactly_the_twelve_fixture_files() -> None:
    found = {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*.md")}
    assert found == set(CODE_WORDS)


@pytest.mark.parametrize("path", sorted(CODE_WORDS))
def test_fixture_length_and_code_word(path: str) -> None:
    text = (CORPUS / path).read_text(encoding="utf-8")
    assert 150 <= len(text.split()) <= 300
    for other_path, word in CODE_WORDS.items():
        assert text.count(word) == (1 if other_path == path else 0)


def test_readable_sets_match_the_spec_counts() -> None:
    assert {key: len(paths) for key, paths in READABLE.items()} == {
        "dev-ana": 8,
        "dev-ben": 9,
        "dev-cal": 0,
    }
