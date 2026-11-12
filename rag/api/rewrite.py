import asyncio
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from rag.common import pg
from rag.common.config import Settings
from rag.common.log import log
from rag.common.models import LLM

REWRITE_SYSTEM = (Path(__file__).parents[2] / "prompts" / "rewrite_v1.txt").read_text(
    encoding="utf-8"
)


def one_line(text: str) -> str:
    return " ".join(text.split())


def rewrite_prompt(turns: list[tuple[str, str]], question: str) -> str:
    lines = [f"{role}: {one_line(text)}" for role, text in turns]
    return "\n".join([*lines, f"question: {one_line(question)}"])


async def rewrite(
    pool: AsyncConnectionPool,
    llm: LLM,
    settings: Settings,
    user_id: str,
    session_id: str,
    question: str,
    degraded: list[str],
    request_id: str,
) -> str | None:
    async with pool.connection() as conn:
        turns = await pg.recent_turns(conn, session_id, user_id, settings.history_turns)
    if not turns:
        return None
    try:
        async with asyncio.timeout(settings.llm_timeout_s):
            rewritten = await llm.complete(
                REWRITE_SYSTEM,
                rewrite_prompt(turns, question),
                model=settings.llm_small_model or "",
                max_tokens=settings.rewrite_max_tokens,
            )
        if not rewritten.strip():
            raise ValueError("the rewriter returned an empty question")
    except Exception as exc:
        log("rewrite_failed", request_id=request_id, error=repr(exc))
        degraded.append("rewrite")
        return None
    return one_line(rewritten)
