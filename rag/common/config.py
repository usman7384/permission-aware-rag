from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    corpus_dir: str = "./corpus"
    index_version: str = "v1"
    store_prefix: str = ""
    model_backend: Literal["real", "fake"] = "real"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_dim: int = 384

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_small_model: str | None = None

    parent_tokens: int = 1024
    child_tokens: int = 128
    contextual_chunking: bool = False
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60
    fused_keep: int = 50
    rerank_keep: int = 10
    abstain_threshold: float = 0.0
    context_token_budget: int = 6000
    answer_max_tokens: int = 1024
    semantic_cache_threshold: float = 0.96
    answer_ttl_s: int = 3600
    chunk_ttl_s: int = 86400
    history_turns: int = 4
    max_question_chars: int = 2000
    rate_limit_per_min: int = 60

    retrieve_timeout_ms: int = 500
    rerank_timeout_ms: int = 2000
    llm_timeout_s: float = 30
    store_timeout_s: float = 5
    fault_slow_ms: int = 5000
    max_deliveries: int = 3
    reclaim_idle_ms: int = 30000
    worker_block_ms: int = 2000

    pg_dsn: str = "postgresql://rag:rag@127.0.0.1:5432/rag"
    redis_url: str = "redis://127.0.0.1:6380/0"
    qdrant_url: str = "http://127.0.0.1:6333"
    opensearch_url: str = "http://127.0.0.1:9200"

    @property
    def chunks_name(self) -> str:
        return f"{self.store_prefix}chunks_{self.index_version}"

    @property
    def answer_cache_name(self) -> str:
        return f"{self.store_prefix}answer_cache_{self.index_version}"
