import asyncio

import pytest
from fastapi import HTTPException

from rag.inference.main import Gate


async def test_gate_admits_concurrency_plus_queue_and_refuses_the_rest() -> None:
    gate, release = Gate(concurrency=1, queue_max=1), asyncio.Event()

    async def hold() -> None:
        async with gate.enter():
            await release.wait()

    holders = [asyncio.create_task(hold()) for _ in range(2)]
    await asyncio.sleep(0)
    assert gate.admitted == 2
    with pytest.raises(HTTPException) as refused:
        async with gate.enter():
            pass
    assert refused.value.status_code == 503 and gate.rejected == 1
    release.set()
    await asyncio.gather(*holders)
    assert gate.admitted == 0
