import uuid

from rag.common.permissions import folder_of, parent_tuples

DOC = uuid.UUID(int=1)


def test_folder_of() -> None:
    assert folder_of("specs/a.md") == "specs"
    assert folder_of("a/b/c.md") == "a/b"
    assert folder_of("top.md") == "_root"


def test_parent_tuples_follow_nested_folders() -> None:
    assert parent_tuples("specs/a.md", DOC) == [(f"doc:{DOC}", "parent", "folder:specs")]
    assert parent_tuples("top.md", DOC) == [(f"doc:{DOC}", "parent", "folder:_root")]
    assert parent_tuples("a/b/c.md", DOC) == [
        (f"doc:{DOC}", "parent", "folder:a/b"),
        ("folder:a/b", "parent", "folder:a"),
    ]
