import asyncio
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.api.hydrate import ParentHit
from rag.common.config import Settings
from rag.common.log import log
from rag.common.models import LLM
from rag.common.text import PII, sentences

ANSWER_SYSTEM = (Path(__file__).parents[2] / "prompts" / "answer_v1.txt").read_text(
    encoding="utf-8"
)
Event = tuple[str, dict[str, Any]]
ABSTAIN_TEXT = "I don't know based on the documents I can access."
_CITATION = re.compile(r"\s*\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?:[.!?]+\s+|\n+)\Z")


@dataclass(frozen=True)
class Context:
    parents: list[ParentHit]
    user: str


def both_ends(ranked: list[ParentHit]) -> list[ParentHit]:
    """Best first, second best last, third second, fourth second-to-last, and so on."""
    return ranked[0::2] + ranked[1::2][::-1]


def assemble(
    parents: list[ParentHit],
    reranked: bool,
    settings: Settings,
    count_tokens: Callable[[str], int],
    question: str,
) -> Context | None:
    """The prompt context, or None to abstain."""
    kept = sorted(parents, key=lambda p: -p.score)
    if not kept or (reranked and kept[0].score < settings.abstain_threshold):
        return None
    sizes = [count_tokens(p.text) for p in kept]
    while kept and sum(sizes) > settings.context_token_budget:
        kept.pop()
        sizes.pop()
    if not kept:
        return None
    ordered = both_ends(kept)
    blocks = "\n\n".join(
        f'<doc n="{n}" path="{p.path}">\n{p.text}\n</doc>' for n, p in enumerate(ordered, start=1)
    )
    return Context(parents=ordered, user=f"{blocks}\n\nQuestion: {question}")


def extractive_fallback(parents: list[ParentHit]) -> str:
    """The first two sentences of each of the top three parents, with its citation marker."""
    top = sorted(enumerate(parents, start=1), key=lambda pair: -pair[1].score)[:3]
    lines = []
    for n, parent in top:
        stripped = (s.strip() for s in sentences(parent.text))
        first = [s for s in stripped if s and not s.startswith("#")][:2]
        lines.append(f"{' '.join(first)} [{n}]\n")
    return "".join(lines)


async def generate(
    llm: LLM, settings: Settings, context: Context, degraded: list[str], request_id: str
) -> AsyncIterator[str]:
    """Stream the answer; on LLM failure add `llm` to `degraded` and, if nothing came yet,
    yield the extractive fallback."""
    received = False
    try:
        async with asyncio.timeout(settings.llm_timeout_s):
            async for piece in llm.stream(
                ANSWER_SYSTEM,
                context.user,
                model=settings.llm_model or "",
                max_tokens=settings.answer_max_tokens,
            ):
                received = True
                yield piece
    except Exception as exc:
        log("llm_failed", request_id=request_id, error=repr(exc), received=received)
        degraded.append("llm")
        if not received:
            yield extractive_fallback(context.parents)


def clean_sentence(sentence: str, n_blocks: int, context_text: str) -> str:
    """Drop citation markers outside 1..n_blocks and redact PII that the context lacks."""

    def citation(match: re.Match[str]) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= n_blocks else ""

    def pii(match: re.Match[str]) -> str:
        return match.group(0) if match.group(0) in context_text else "[redacted]"

    return PII.sub(pii, _CITATION.sub(citation, sentence))


async def output_guard(pieces: AsyncIterator[str], parents: list[ParentHit]) -> AsyncIterator[str]:
    """Buffer the stream and release it one cleaned sentence at a time."""
    context_text = "\n".join(p.text for p in parents)
    buffer = ""
    async for piece in pieces:
        buffer += piece
        done = sentences(buffer)
        buffer = done.pop() if done and not _SENTENCE_END.search(done[-1]) else ""
        for sentence in done:
            if cleaned := clean_sentence(sentence, len(parents), context_text):
                yield cleaned
    if buffer and (cleaned := clean_sentence(buffer, len(parents), context_text)):
        yield cleaned


def _citations(answer: str, parents: list[ParentHit]) -> list[dict[str, Any]]:
    """The context blocks cited in a guarded answer, by block number."""
    numbers = sorted({int(n) for n in _CITATION.findall(answer)})
    return [
        {
            "n": n,
            "doc_id": parents[n - 1].doc_id,
            "path": parents[n - 1].path,
            "parent_id": parents[n - 1].parent_id,
        }
        for n in numbers
    ]


def closing_events(answer: str, parents: list[ParentHit]) -> list[Event]:
    """The `citations` event, then an `uncited` warning if the answer cites no block."""
    items = _citations(answer, parents)
    events: list[Event] = [("citations", {"items": items})]
    if not items:
        events.append(("warning", {"code": "uncited"}))
    return events
