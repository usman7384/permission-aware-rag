import hashlib
import json

from rag.common.text import normalize_query


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(principals: list[str]) -> str:
    return _sha256(",".join(sorted(principals)))


def cache_key(question: str, fingerprint: str, index_version: str) -> str:
    return _sha256(f"{normalize_query(question)}|{fingerprint}|{index_version}")


def api_key_hash(api_key: str) -> str:
    return _sha256(api_key)


JOBS_STREAM = "ingest:jobs"
JOBS_GROUP = "workers"
DEAD_STREAM = "ingest:dead"
CHANGES_STREAM = "events:changes"
CHANGES_GROUP = "invalidators"


def doc_event(event_type: str, doc_ids: list[str]) -> dict[str, str]:
    return {"type": event_type, "doc_ids": json.dumps(doc_ids)}


def tuple_event(subject: str, relation: str, obj: str) -> dict[str, str]:
    return {"type": "tuple_changed", "subject": subject, "relation": relation, "object": obj}


def answer_key(cache_key: str) -> str:
    return f"ans:{cache_key}"


def chunk_key(parent_id: str) -> str:
    return f"chunk:{parent_id}"


def doc_answers_key(doc_id: str) -> str:
    return f"doc2ans:{doc_id}"


def rate_limit_key(key_hash: str, epoch_minute: int) -> str:
    return f"rl:{key_hash}:{epoch_minute}"


def fault_key(component: str) -> str:
    return f"fault:{component}"
