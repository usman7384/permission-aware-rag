import re
from collections.abc import Callable
from typing import NamedTuple

from rag.worker.parse import Section, paragraphs

_SENTENCE = re.compile(r".+?(?:[.!?]+\s+|\n+|\Z)", re.DOTALL)
_WORD = re.compile(r"\s*\S+\s*")

Counter = Callable[[str], int]


class Parent(NamedTuple):
    heading: str | None
    text: str
    children: list[str]


def sentences(text: str) -> list[str]:
    return _SENTENCE.findall(text)


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def _pack(pieces: list[str], limit: int, count: Counter) -> list[list[str]]:
    """Group consecutive pieces greedily so each group's text has at most `limit` tokens."""
    groups: list[list[str]] = []
    for piece in pieces:
        if groups and count("".join(groups[-1]) + piece) <= limit:
            groups[-1].append(piece)
        else:
            groups.append([piece])
    return groups


def _fit(
    text: str, limit: int, count: Counter, splitters: list[Callable[[str], list[str]]]
) -> list[str]:
    """Split `text` with the coarsest splitter that works into pieces of at most `limit` tokens."""
    if count(text) <= limit or not splitters:
        return [text]
    pieces = [p for piece in splitters[0](text) for p in _fit(piece, limit, count, splitters[1:])]
    return ["".join(group) for group in _pack(pieces, limit, count)]


def chunk(
    sections: list[Section], parent_tokens: int, child_tokens: int, count: Counter
) -> list[Parent]:
    """Pack sections into parents and split each parent into children; no text is lost."""
    units = [
        (section.heading, piece)
        for section in sections
        for piece in _fit(section.text, parent_tokens, count, [paragraphs, sentences, words])
    ]
    parents = []
    start = 0
    for group in _pack([text for _, text in units], parent_tokens, count):
        text = "".join(group)
        children = _fit(text, child_tokens, count, [sentences, words])
        parents.append(Parent(units[start][0], text, children))
        start += len(group)
    return parents
