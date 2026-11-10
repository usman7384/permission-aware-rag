from pathlib import Path

import pytest

from rag.common.models import FakeEmbedder
from rag.worker.chunk import Parent, chunk
from rag.worker.parse import Section, parse

CORPUS = Path(__file__).parents[1] / "fixtures" / "corpus"
count = FakeEmbedder(384).count_tokens


def check(parents, parent_tokens: int, child_tokens: int) -> None:
    for parent in parents:
        assert count(parent.text) <= parent_tokens
        assert all(count(child) <= child_tokens for child in parent.children)
        assert "".join(parent.children) == parent.text


FIXTURE_PATHS = sorted(p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*.md"))


@pytest.mark.parametrize("path", FIXTURE_PATHS)
def test_fixture_files_respect_limits(path: str) -> None:
    sections = parse(path, (CORPUS / path).read_bytes())
    parents = chunk(sections, parent_tokens=60, child_tokens=16, count=count)
    check(parents, 60, 16)
    assert "".join(p.text for p in parents) == "".join(s.text for s in sections)


def test_oversized_section_splits_on_paragraphs() -> None:
    paragraph = "One two three four five. Six seven eight nine ten.\n\n"
    parents = chunk([Section("Big", "# Big\n\n" + paragraph * 4)], 25, 6, count)
    check(parents, 25, 6)
    assert [p.heading for p in parents] == ["Big", "Big"]
    assert parents[1].text == paragraph * 2


def test_small_sections_pack_into_one_parent() -> None:
    sections = [Section("A", "# A\nalpha beta.\n"), Section("B", "# B\ngamma delta.\n")]
    parents = chunk(sections, 100, 100, count)
    text = "# A\nalpha beta.\n# B\ngamma delta.\n"
    assert parents == [Parent("A", text, [text])]


def test_long_sentence_splits_on_words() -> None:
    text = " ".join(f"w{i}" for i in range(30)) + "."
    parents = chunk([Section(None, text)], 100, 7, count)
    check(parents, 100, 7)
    assert len(parents[0].children) == 5
