import io
import re
from collections.abc import Callable
from functools import cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_FENCE = ("```", "~~~")


class Section(NamedTuple):
    heading: str | None
    text: str


def _split_before(text: str, starts_piece: Callable[[str, str], bool]) -> list[str]:
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
    return _split_before(text, lambda line, previous: bool(line.strip()) and not previous.strip())


def _markdown_sections(text: str) -> list[Section]:
    pieces = _split_before(text, lambda line, _: _HEADING.match(line) is not None)
    sections = []
    for piece in pieces:
        match = _HEADING.match(piece.splitlines()[0])
        if match or piece.strip():
            sections.append(Section(match.group(1) if match else None, piece))
    return sections


@cache
def _pdf_converter() -> "DocumentConverter":
    # Lazy import: Markdown-only work never loads docling.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def pdf_to_markdown(name: str, data: bytes) -> str:
    from docling.datamodel.base_models import DocumentStream

    stream = DocumentStream(name=PurePosixPath(name).name, stream=io.BytesIO(data))
    return _pdf_converter().convert(stream).document.export_to_markdown()


def parse(name: str, data: bytes) -> list[Section]:
    if name.endswith(".pdf"):
        return _markdown_sections(pdf_to_markdown(name, data))
    text = data.decode("utf-8")
    if name.endswith(".md"):
        return _markdown_sections(text)
    return [Section(None, text)] if text.strip() else []
