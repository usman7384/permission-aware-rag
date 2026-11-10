import asyncio

import pytest

from rag.api import retrieve as retrieve_module
from rag.api.retrieve import Candidate, RetrievalUnavailable, retrieve, sparse
from rag.common.ids import doc_id
from rag.common.models import FakeEmbedder
from rag.worker.index import Deps

ANA = ["group:eng", "group:staff", "user:ana"]
API_LIMITS = str(doc_id("specs/api-limits.md"))


async def run_retrieve(deps: Deps, question: str) -> tuple[list[list[Candidate]], list[str]]:
    vector = FakeEmbedder(deps.settings.embed_dim).embed_query(question)
    return await retrieve(
        deps.qdrant, deps.opensearch, deps.settings, question, vector, ANA, "test"
    )


async def test_code_word_is_found_by_the_sparse_branch_alone(ingested: Deps) -> None:
    hits = await sparse(ingested.opensearch, ingested.settings, "ZEBRA-7", ANA)
    assert hits[0].doc_id == API_LIMITS
    assert hits[0].version == 1


@pytest.mark.parametrize("slow", ["dense", "sparse"])
async def test_timed_out_branch_is_degraded_and_the_other_answers(
    ingested: Deps, monkeypatch: pytest.MonkeyPatch, slow: str
) -> None:
    async def hang(*args: object) -> list[Candidate]:
        await asyncio.sleep(ingested.settings.retrieve_timeout_ms / 1000 * 4)
        return []

    monkeypatch.setattr(retrieve_module, slow, hang)
    branches, degraded = await run_retrieve(ingested, "ZEBRA-7")
    assert degraded == [slow]
    assert len(branches) == 1
    assert API_LIMITS in {c.doc_id for c in branches[0]}


async def test_both_branches_failing_raises(
    ingested: Deps, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*args: object) -> list[Candidate]:
        raise ConnectionError("store down")

    monkeypatch.setattr(retrieve_module, "dense", fail)
    monkeypatch.setattr(retrieve_module, "sparse", fail)
    with pytest.raises(RetrievalUnavailable):
        await run_retrieve(ingested, "ZEBRA-7")
