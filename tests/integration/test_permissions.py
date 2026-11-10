import pytest

from rag.common.ids import doc_id
from rag.common.permissions import document_readers, expand_principals, readable_docs
from rag.worker.index import Deps


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("user:ana", ["group:eng", "group:staff", "user:ana"]),
        ("user:ben", ["group:hr", "group:staff", "user:ben"]),
        ("user:cal", ["user:cal"]),
    ],
)
async def test_expand_principals_matches_the_fixture_groups(
    deps: Deps, user: str, expected: list[str]
) -> None:
    async with deps.pg.connection() as conn:
        assert await expand_principals(conn, user) == expected


async def test_document_readers_of_api_limits(ingested: Deps) -> None:
    async with ingested.pg.connection() as conn:
        readers = await document_readers(conn, doc_id("specs/api-limits.md"))
    assert readers == ["group:eng", "user:ben"]


async def test_readable_docs_follows_folders_and_direct_viewers(ingested: Deps) -> None:
    docs = {path: str(doc_id(path)) for path in ("specs/api-limits.md", "specs/auth-design.md")}
    docs["hr/expenses.md"] = str(doc_id("hr/expenses.md"))
    docs["public/glossary.md"] = str(doc_id("public/glossary.md"))
    ben = ["group:hr", "group:staff", "user:ben"]
    async with ingested.pg.connection() as conn:
        readable = await readable_docs(conn, ben, list(docs.values()))
        nothing = await readable_docs(conn, ["user:cal"], list(docs.values()))
    expected = {docs["specs/api-limits.md"], docs["hr/expenses.md"], docs["public/glossary.md"]}
    assert readable == expected
    assert nothing == set()
