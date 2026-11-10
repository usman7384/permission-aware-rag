from rag.common.ids import chunk_id, doc_id, parent_id


def test_doc_id_is_deterministic_and_differs_by_path() -> None:
    assert doc_id("specs/api-limits.md") == doc_id("specs/api-limits.md")
    assert doc_id("specs/api-limits.md") != doc_id("specs/auth-design.md")


def test_parent_id_differs_by_version_and_ordinal() -> None:
    doc = doc_id("specs/api-limits.md")
    assert parent_id(doc, 1, 0) == parent_id(doc, 1, 0)
    assert len({parent_id(doc, 1, 0), parent_id(doc, 2, 0), parent_id(doc, 1, 1)}) == 3
    assert parent_id(doc, 1, 0) != parent_id(doc_id("hr/expenses.md"), 1, 0)


def test_chunk_id_differs_by_version_and_both_ordinals() -> None:
    doc = doc_id("hr/expenses.md")
    assert chunk_id(doc, 1, 0, 0) == chunk_id(doc, 1, 0, 0)
    ids = {
        chunk_id(doc, 1, 0, 0),
        chunk_id(doc, 2, 0, 0),
        chunk_id(doc, 1, 1, 0),
        chunk_id(doc, 1, 0, 1),
    }
    assert len(ids) == 4
    assert chunk_id(doc, 1, 0, 0) != parent_id(doc, 1, 0)
