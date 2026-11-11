import httpx
from conftest import search_paths
from fixture_corpus import CODE_WORDS, READABLE
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.api import pipeline
from rag.common import redis_
from rag.common.ids import doc_id
from rag.common.keys import CHANGES_GROUP, CHANGES_STREAM
from rag.worker import invalidate, relay
from rag.worker.index import Deps

SPECS = sorted(path for path in CODE_WORDS if path.startswith("specs/"))


async def change_tuple(
    client: httpx.AsyncClient, op: str, subject: str, relation: str, obj: str, key: str
) -> httpx.Response:
    body = {"op": op, "subject": subject, "relation": relation, "object": obj}
    return await client.post("/admin/tuples", json=body, headers={"X-API-Key": key})


async def run_invalidator(deps: Deps) -> None:
    """Relay the outbox to `events:changes`, then handle every event."""
    await redis_.ensure_group(deps.redis, CHANGES_STREAM, CHANGES_GROUP)
    while await relay.relay_once(deps):
        pass
    while await invalidate.consume_once(deps, "test"):
        pass


async def outbox_rows(deps: Deps) -> int:
    async with deps.pg.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM outbox")
        (count,) = await cur.fetchone()
    return count


async def stamped(deps: Deps, path: str) -> set[tuple[str, ...]]:
    """The distinct `principals` payloads of a document's chunks in Qdrant and OpenSearch."""
    doc = str(doc_id(path))
    condition = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc))])
    points, _ = await deps.qdrant.scroll(
        deps.settings.chunks_name, scroll_filter=condition, limit=10_000
    )
    response = await deps.opensearch.search(
        index=deps.settings.chunks_name, body={"query": {"term": {"doc_id": doc}}, "size": 10_000}
    )
    hits = response["hits"]["hits"]
    assert points and hits
    payloads = [p.payload["principals"] for p in points] + [
        h["_source"]["principals"] for h in hits
    ]
    return {tuple(principals) for principals in payloads}


async def visible(deps: pipeline.Deps, api_key: str) -> set[str]:
    """Every document that a search for any fixture code word returns for the key."""
    found: set[str] = set()
    for word in CODE_WORDS.values():
        found |= await search_paths(deps, api_key, word)
    return found


async def test_revoked_folder_is_hidden_before_the_invalidator_runs(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    response = await change_tuple(
        client, "remove", "group:eng", "viewer", "folder:specs", "dev-admin"
    )
    assert response.status_code == 204
    while await relay.relay_once(ingested):
        pass
    [(_, event)] = await ingested.redis.xrevrange(CHANGES_STREAM, count=1)
    assert event == {
        "type": "tuple_changed",
        "subject": "group:eng",
        "relation": "viewer",
        "object": "folder:specs",
    }
    # The chunks still carry the old readers, so only the recheck gate hides them.
    for path in SPECS:
        assert all("group:eng" in principals for principals in await stamped(ingested, path))
    assert await visible(api_deps, "dev-ana") == READABLE["dev-ana"] - set(SPECS)


async def test_invalidator_restamps_the_folder_documents(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    await change_tuple(client, "remove", "group:eng", "viewer", "folder:specs", "dev-admin")
    await run_invalidator(ingested)
    for path in SPECS:
        expected = {("user:ben",)} if path == "specs/api-limits.md" else {()}
        assert await stamped(ingested, path) == expected, path
    assert await stamped(ingested, "public/glossary.md") == {("group:staff",)}
    assert await visible(api_deps, "dev-ben") == READABLE["dev-ben"]


async def test_new_viewer_sees_only_that_document_after_the_restamp(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    expenses = f"doc:{doc_id('hr/expenses.md')}"
    response = await change_tuple(client, "add", "user:cal", "viewer", expenses, "dev-admin")
    assert response.status_code == 204
    assert await visible(api_deps, "dev-cal") == set()
    await run_invalidator(ingested)
    assert await stamped(ingested, "hr/expenses.md") == {("group:hr", "user:cal")}
    assert await visible(api_deps, "dev-cal") == {"hr/expenses.md"}


async def test_leaving_a_group_changes_fingerprint_and_results_without_restamp(
    client: httpx.AsyncClient, api_deps: pipeline.Deps, ingested: Deps
) -> None:
    async with api_deps.pg.connection() as conn:
        before, old_fingerprint = await pipeline.expand(conn, "user:ana")
    stamps = {path: await stamped(ingested, path) for path in CODE_WORDS}
    response = await change_tuple(client, "remove", "user:ana", "member", "group:eng", "dev-admin")
    assert response.status_code == 204
    async with api_deps.pg.connection() as conn:
        after, new_fingerprint = await pipeline.expand(conn, "user:ana")
    assert before == ["group:eng", "group:staff", "user:ana"] and after == ["user:ana"]
    assert new_fingerprint != old_fingerprint
    assert await visible(api_deps, "dev-ana") == set()
    await run_invalidator(ingested)
    assert {path: await stamped(ingested, path) for path in CODE_WORDS} == stamps


async def test_non_admin_key_gets_401_and_changes_nothing(
    client: httpx.AsyncClient, ingested: Deps
) -> None:
    events = await outbox_rows(ingested)
    for key in ("dev-ana", "dev-cal", "not-a-key"):
        response = await change_tuple(client, "add", "user:cal", "viewer", "folder:specs", key)
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"
    response = await client.post(
        "/admin/tuples",
        json={"op": "add", "subject": "user:cal", "relation": "viewer", "object": "folder:specs"},
    )
    assert response.status_code == 401
    async with ingested.pg.connection() as conn:
        cur = await conn.execute("SELECT count(*) FROM tuples WHERE subject = 'user:cal'")
        assert await cur.fetchone() == (0,)
    assert await outbox_rows(ingested) == events
