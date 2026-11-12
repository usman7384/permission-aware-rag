import asyncio
from collections.abc import AsyncIterator

from redis.asyncio import Redis

from rag.common import redis_
from rag.common.models import LLM


class InjectedFault(Exception): ...


async def inject(redis: Redis, component: str, slow_ms: int) -> None:
    mode = await redis_.get_fault(redis, component)
    if mode == "down":
        raise InjectedFault(f"{component} is down (injected fault)")
    if mode == "slow":
        await asyncio.sleep(slow_ms / 1000)


class FaultyLLM:
    def __init__(self, llm: LLM, redis: Redis, slow_ms: int) -> None:
        self.llm = llm
        self.redis = redis
        self.slow_ms = slow_ms

    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        await inject(self.redis, "llm", self.slow_ms)
        return await self.llm.complete(system, user, model=model, max_tokens=max_tokens)

    async def stream(
        self, system: str, user: str, *, model: str, max_tokens: int
    ) -> AsyncIterator[str]:
        await inject(self.redis, "llm", self.slow_ms)
        async for piece in self.llm.stream(system, user, model=model, max_tokens=max_tokens):
            yield piece
