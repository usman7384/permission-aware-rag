import asyncio
from pathlib import Path

from rag.common.config import Settings
from rag.common.models import LLM
from rag.common.text import sentences
from rag.worker.chunk import Counter, Parent, words

CONTEXT_SYSTEM = (Path(__file__).parents[2] / "prompts" / "context_v1.txt").read_text(
    encoding="utf-8"
)


def head(text: str, limit: int, count: Counter) -> str:
    pieces = words(text)
    low, high = 0, len(pieces)
    while low < high:
        middle = (low + high + 1) // 2
        if count("".join(pieces[:middle])) <= limit:
            low = middle
        else:
            high = middle - 1
    return "".join(pieces[:low])


def context_prompt(document_head: str, parent_text: str) -> str:
    return f"<document>\n{document_head}\n</document>\n<chunk>\n{parent_text}\n</chunk>"


def first_two_sentences(reply: str) -> str:
    return " ".join(" ".join(sentences(reply.strip())[:2]).split())


async def contextualize(
    llm: LLM, settings: Settings, document: str, parents: list[Parent], count: Counter
) -> list[str]:
    document_head = head(document, settings.context_doc_tokens, count)

    async def one(parent: Parent) -> str:
        async with asyncio.timeout(settings.llm_timeout_s):
            reply = await llm.complete(
                CONTEXT_SYSTEM,
                context_prompt(document_head, parent.text),
                model=settings.llm_small_model or "",
                max_tokens=settings.context_max_tokens,
            )
        context = first_two_sentences(reply)
        if not context:
            raise ValueError("the contextualizer returned an empty context")
        return context

    return await asyncio.gather(*(one(parent) for parent in parents))
