import hashlib
import json
import math
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

import httpx

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
    def __init__(self, model_name: str, query_prefix: str) -> None:
        # Lazy import: the fake backend never loads torch.
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


def local_embedder(settings: Settings) -> Embedder:
    if settings.model_backend == "fake":
        return FakeEmbedder(settings.embed_dim)
    return SentenceTransformerEmbedder(settings.embed_model, settings.embed_query_prefix)


class InferenceUnavailable(Exception): ...


def post_inference(client: httpx.Client, path: str, body: dict[str, Any]) -> Any:
    try:
        response = client.post(path, json=body)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise InferenceUnavailable(f"{path}: {exc!r}") from exc
    return response.json()


def token_counter(settings: Settings) -> Callable[[str], int]:
    if settings.model_backend == "fake":
        return FakeEmbedder(settings.embed_dim).count_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.embed_model)
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


class RemoteEmbedder:
    def __init__(self, client: httpx.Client, dim: int, count_tokens: Callable[[str], int]) -> None:
        self.client = client
        self.dim = dim
        self.count_tokens = count_tokens

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        body = {"texts": texts, "kind": "document"}
        return post_inference(self.client, "/embed", body)["vectors"]

    def embed_query(self, text: str) -> list[float]:
        body = {"texts": [text], "kind": "query"}
        return post_inference(self.client, "/embed", body)["vectors"][0]


def inference_client(url: str, timeout_ms: int) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=timeout_ms / 1000)


def make_embedder(settings: Settings) -> Embedder:
    if not settings.inference_url:
        return local_embedder(settings)
    client = inference_client(settings.inference_url, settings.inference_timeout_ms)
    return RemoteEmbedder(client, settings.embed_dim, token_counter(settings))


class FakeReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_tokens = set(tokens(query))
        return [float(len(query_tokens & set(tokens(p)))) for p in passages]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        # Lazy import: the fake backend never loads torch.
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, activation_fn=torch.nn.Identity())

    def score(self, query: str, passages: list[str]) -> list[float]:
        return self.model.predict([(query, p) for p in passages]).tolist()


def local_reranker(settings: Settings) -> Reranker:
    if settings.model_backend == "fake":
        return FakeReranker()
    return CrossEncoderReranker(settings.rerank_model)


class RemoteReranker:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def score(self, query: str, passages: list[str]) -> list[float]:
        body = {"query": query, "passages": passages}
        return post_inference(self.client, "/rerank", body)["scores"]


def make_reranker(settings: Settings) -> Reranker:
    if not settings.inference_url:
        return local_reranker(settings)
    return RemoteReranker(inference_client(settings.inference_url, settings.rerank_timeout_ms))


def _rewrite(user: str) -> str:
    lines = user.splitlines()
    previous = [line.removeprefix("user: ") for line in lines if line.startswith("user: ")]
    question = next(
        line.removeprefix("question: ") for line in lines if line.startswith("question: ")
    )
    return f"{previous[-1]} {question}" if previous else question


class FakeLLM:
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


class ChatCompletionsLLM:
    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_s = timeout_s

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, headers=self.headers, timeout=self.timeout_s
        )

    def _body(self, system: str, user: str, model: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }

    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        async with self._client() as client:
            response = await client.post(
                "chat/completions", json=self._body(system, user, model, max_tokens)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def stream(
        self, system: str, user: str, *, model: str, max_tokens: int
    ) -> AsyncIterator[str]:
        body = {**self._body(system, user, model, max_tokens), "stream": True}
        async with (
            self._client() as client,
            client.stream("POST", "chat/completions", json=body) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                data = line.removeprefix("data:").strip()
                if not line.startswith("data:") or data == "[DONE]":
                    continue
                choices = json.loads(data)["choices"]
                if choices and choices[0]["delta"].get("content"):
                    yield choices[0]["delta"]["content"]


def make_llm(settings: Settings) -> LLM:
    if settings.model_backend == "fake":
        return FakeLLM()
    return ChatCompletionsLLM(
        settings.llm_base_url or "", settings.llm_api_key or "", settings.llm_timeout_s
    )
