import re
from collections.abc import Callable
from typing import NamedTuple

_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_FENCE = ("```", "~~~")


class Section(NamedTuple):
    heading: str | None
    text: str


def _split_before(text: str, starts_piece: Callable[[str, str], bool]) -> list[str]:
    """Cut `text` before each line where `starts_piece(line, previous_line)`, never in a fence."""
    pieces: list[str] = []
    in_fence = False
    previous = ""
    for line in text.splitlines(keepends=True):
        if not in_fence and pieces and starts_piece(line, previous):
            pieces.append("")
        if not pieces:
            pieces.append("")
        pieces[-1] += line
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
        previous = line
    return pieces


def paragraphs(text: str) -> list[str]:
    """Split on blank lines; a fenced code block or a table stays in one piece."""
    return _split_before(text, lambda line, previous: bool(line.strip()) and not previous.strip())


def _markdown_sections(text: str) -> list[Section]:
    pieces = _split_before(text, lambda line, _: _HEADING.match(line) is not None)
    sections = []
    for piece in pieces:
        match = _HEADING.match(piece.splitlines()[0])
        if match or piece.strip():
            sections.append(Section(match.group(1) if match else None, piece))
    return sections


def parse(name: str, data: bytes) -> list[Section]:
    """Sections of a file; their texts concatenate back to the file, minus blank preambles."""
    text = data.decode("utf-8")
    if name.endswith(".md"):
        return _markdown_sections(text)
    return [Section(None, text)] if text.strip() else []
