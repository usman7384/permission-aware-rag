import uuid


def doc_id(path: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, path)


def parent_id(doc: uuid.UUID, version: int, parent_ordinal: int) -> uuid.UUID:
    return uuid.uuid5(doc, f"{version}:{parent_ordinal}")


def chunk_id(doc: uuid.UUID, version: int, parent_ordinal: int, child_ordinal: int) -> uuid.UUID:
    return uuid.uuid5(doc, f"{version}:{parent_ordinal}:{child_ordinal}")


def answer_point_id(cache_key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, cache_key)
