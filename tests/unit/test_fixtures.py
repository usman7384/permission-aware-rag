from pathlib import Path

import pytest

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"
CODE_WORDS = {
    "specs/api-limits.md": "ZEBRA-7",
    "specs/deploy-process.md": "OTTER-3",
    "specs/auth-design.md": "HERON-9",
    "specs/storage-layout.md": "LYNX-2",
    "hr/leave-policy.md": "MAPLE-4",
    "hr/expenses.md": "CEDAR-8",
    "hr/onboarding.md": "BIRCH-1",
    "hr/remote-work.md": "ASPEN-6",
    "public/company-values.md": "CORAL-5",
    "public/office-hours.md": "PEARL-0",
    "public/glossary.md": "AMBER-2",
    "public/holidays.md": "SLATE-3",
}


def test_corpus_has_exactly_the_twelve_fixture_files() -> None:
    found = {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*.md")}
    assert found == set(CODE_WORDS)


@pytest.mark.parametrize("path", sorted(CODE_WORDS))
def test_fixture_length_and_code_word(path: str) -> None:
    text = (CORPUS / path).read_text(encoding="utf-8")
    assert 150 <= len(text.split()) <= 300
    for other_path, word in CODE_WORDS.items():
        assert text.count(word) == (1 if other_path == path else 0)
