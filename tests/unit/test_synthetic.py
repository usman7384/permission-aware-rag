import pytest

from rag.evals.synthetic import synthetic_acl


def _readable(acl, user: str) -> int:
    """Documents the user reads, by expanding the generated tuples as section 7.1 does."""
    principals, frontier = {user}, [user]
    while frontier:
        subject = frontier.pop()
        for s, relation, obj in acl.member_tuples:
            if s == subject and relation == "member" and obj not in principals:
                principals.add(obj)
                frontier.append(obj)
    folders = {obj for s, _, obj in acl.viewer_tuples if s in principals}
    return sum(f"folder:{folder}" in folders for folder in acl.folders)


@pytest.mark.parametrize("selectivity", [1.0, 0.5, 0.1, 0.01, 0.001])
def test_tuples_give_the_user_the_target_share_of_documents(selectivity: float) -> None:
    acl = synthetic_acl("scifact", 5183, 1000, selectivity, "user:bench")
    readable = _readable(acl, "user:bench")
    assert readable / 5183 == acl.selectivity
    assert abs(acl.selectivity - selectivity) <= 1 / 1000


def test_documents_are_spread_evenly_and_other_users_read_nothing() -> None:
    acl = synthetic_acl("scifact", 10, 3, 0.5, "user:bench")
    assert acl.folders == ["scifact/f0000", "scifact/f0001", "scifact/f0002"] * 3 + [
        "scifact/f0000"
    ]
    assert acl.selectivity == 0.4
    assert _readable(acl, "user:other") == 0


def test_impossible_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        synthetic_acl("d", 10, 3, 0.0, "user:bench")
    with pytest.raises(ValueError):
        synthetic_acl("d", 10, 11, 0.5, "user:bench")
