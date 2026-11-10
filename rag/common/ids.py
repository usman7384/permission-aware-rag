import uuid


def doc_id(path: str) -> uuid.UUID:
    """ID for a document; `path` is relative to CORPUS_DIR with `/` separators."""
    return uuid.uuid5(uuid.NAMESPACE_URL, path)


def parent_id(doc: uuid.UUID, version: int, parent_ordinal: int) -> uuid.UUID:
    return uuid.uuid5(doc, f"{version}:{parent_ordinal}")


def chunk_id(doc: uuid.UUID, version: int, parent_ordinal: int, child_ordinal: int) -> uuid.UUID:
    return uuid.uuid5(doc, f"{version}:{parent_ordinal}:{child_ordinal}")
