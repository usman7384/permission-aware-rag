import hashlib
import math
from collections.abc import AsyncIterator
from typing import Protocol

from rag.common.config import Settings
from rag.common.text import tokens


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class LLM(Protocol):
    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str: ...

    def stream(
        self, system: str, user: str, *, model: str, max_tokens: int
    ) -> AsyncIterator[str]: ...


class FakeEmbedder:
    """Hashed bag of words, L2-normalized."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % self.dim

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokens(text):
            vector[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        return len(tokens(text))


class SentenceTransformerEmbedder:
    """A sentence-transformers model with L2-normalized output and a query prefix."""

    def __init__(self, model_name: str, query_prefix: str) -> None:
        # Imported here so the fake backend never loads torch.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])


def make_embedder(settings: Settings) -> Embedder:
    if settings.model_backend == "fake":
        return FakeEmbedder(settings.embed_dim)
    return SentenceTransformerEmbedder(settings.embed_model, settings.embed_query_prefix)


class FakeReranker:
    """Score is the number of distinct query tokens present in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokens(query))
        return [float(len(query_tokens & set(tokens(p)))) for p in passages]


def _rewrite(user: str) -> str:
    """Join the last `user:` turn line and the `question:` line of a rewrite prompt."""
    lines = user.splitlines()
    previous = [line.removeprefix("user: ") for line in lines if line.startswith("user: ")]
    question = next(
        line.removeprefix("question: ") for line in lines if line.startswith("question: ")
    )
    return f"{previous[-1]} {question}" if previous else question


class FakeLLM:
    """Replies by the prompt kind named on the first line of `system`."""

    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        kind = system.splitlines()[0].removeprefix("KIND:").strip()
        if kind == "answer":
            return "Based on the documents, the answer is in [1]."
        if kind == "rewrite":
            return _rewrite(user)
        if kind == "context":
            return "This chunk is from the document."
        if kind == "judge":
            return "1"
        raise ValueError(f"unknown prompt kind: {kind!r}")

    async def stream(
        self, system: str, user: str, *, model: str, max_tokens: int
    ) -> AsyncIterator[str]:
        text = await self.complete(system, user, model=model, max_tokens=max_tokens)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
