from rag.common.keys import cache_key, fingerprint


def test_fingerprint_ignores_principal_order() -> None:
    assert fingerprint(["user:ana", "group:eng", "group:staff"]) == fingerprint(
        ["group:staff", "user:ana", "group:eng"]
    )
    assert fingerprint(["user:ana", "group:eng"]) != fingerprint(["user:ana"])


def test_cache_key_changes_with_query_fingerprint_and_index_version() -> None:
    fp = fingerprint(["user:ana", "group:eng"])
    base = cache_key("What is ZEBRA-7?", fp, "v1")
    assert cache_key("What is ZEBRA-7?", fp, "v1") == base
    assert cache_key("What is OTTER-3?", fp, "v1") != base
    assert cache_key("What is ZEBRA-7?", fingerprint(["user:ben"]), "v1") != base
    assert cache_key("What is ZEBRA-7?", fp, "v2") != base


def test_cache_key_uses_the_normalized_query() -> None:
    fp = fingerprint(["user:ana"])
    assert cache_key("  what is zebra-7 ", fp, "v1") == cache_key("What is ZEBRA-7?", fp, "v1")
