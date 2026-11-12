from rag.api.auth import ApiError
from rag.common.text import INJECTION, PII


def input_guard(question: str, max_chars: int) -> None:
    if len(question) > max_chars:
        raise ApiError(400, "too_long", f"the question is longer than {max_chars} characters")
    if INJECTION.search(question):
        raise ApiError(400, "injection", "the question looks like a prompt injection")
    if PII.search(question):
        raise ApiError(400, "pii", "the question contains an email address or a long number")
