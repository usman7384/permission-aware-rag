import pytest
from test_ingest import fetch, ingest

from rag.worker.index import Deps


async def indexed_texts(deps: Deps) -> dict[str, tuple[str, str]]:
    """chunk_id -> (parent_id, indexed text) from OpenSearch."""
    response = await deps.opensearch.search(
        index=deps.settings.chunks_name, body={"query": {"match_all": {}}, "size": 10_000}
    )
    return {
        hit["_id"]: (hit["_source"]["parent_id"], hit["_source"]["text"])
        for hit in response["hits"]["hits"]
    }


async def dense_vectors(deps: Deps) -> dict[str, list[float]]:
    points, _ = await deps.qdrant.scroll(deps.settings.chunks_name, limit=10_000, with_vectors=True)
    return {str(point.id): point.vector for point in points}


@pytest.mark.parametrize("enabled", [True, False])
async def test_contextual_chunking_prefixes_children_only_when_on(
    deps: Deps, enabled: bool
) -> None:
    deps.settings = deps.settings.model_copy(update={"contextual_chunking": enabled})
    await ingest(deps)
    rows = await fetch(deps, "SELECT parent_id::text, context, text FROM parents")
    parents = {parent: (context, text) for parent, context, text in rows}
    assert {context is not None for context, _ in parents.values()} == {enabled}
    texts = await indexed_texts(deps)
    assert texts
    for parent, text in texts.values():
        context, parent_text = parents[parent]
        if enabled:
            assert context == "This chunk is from the document."
            assert text.startswith(context + "\n")
            assert text.removeprefix(context + "\n") in parent_text
        else:
            assert context is None
            assert text in parent_text
    vectors = await dense_vectors(deps)
    for chunk, (_, text) in texts.items():
        assert vectors[chunk] == pytest.approx(deps.embedder.embed_query(text))
