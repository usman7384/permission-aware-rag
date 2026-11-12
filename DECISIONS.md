# DECISIONS.md

1. **Ports.** Postgres 5432, Redis 6380 on the host (6379 inside Compose; host 6379 was taken), Qdrant 6333/6334, OpenSearch 9200, API 8000, inference 8100 (127.0.0.1 only), test inference 8101; clients use `127.0.0.1` because `localhost` resolved to IPv6 and hung.
2. **Embedder.** `BAAI/bge-small-en-v1.5` (384 dims) with the query prefix from its model card; chunk sizes are counted with its tokenizer.
3. **Reranker.** `cross-encoder/ms-marco-MiniLM-L-6-v2`, returning raw logits; on failure or timeout the fused order is kept and `degraded: rerank` is set.
4. **Other models.** PDFs are parsed with `docling`; Linux images use CPU-only torch wheels.
5. **Index version.** Change `INDEX_VERSION` and run `make migrate` after changing models, chunking, `CONTEXTUAL_CHUNKING` or Qdrant quantization; the hash check alone does not re-index.
6. **Semantic cache threshold.** `SEMANTIC_CACHE_THRESHOLD=0.98` (owner decision); tests using the fake embedder set 0.8.
7. **Outbox.** Every `events:changes` event is written in the same Postgres transaction as its change; the relay publishes in id order with `FOR UPDATE SKIP LOCKED`, at least once, so every invalidator action is idempotent.
8. **Filtered search.** An exact Qdrant `count` with the principals filter picks an exact scan at or below `EXACT_SCAN_MAX_CHUNKS` (2000), HNSW above; the plan is logged as `dense_plan`.
9. **Inference limits.** `INFERENCE_CONCURRENCY=2` model calls and `INFERENCE_QUEUE_MAX=8` waiting, then 503 at once; `/embed` times out after `INFERENCE_TIMEOUT_MS`, `/rerank` after `RERANK_TIMEOUT_MS`; an embed failure on the query path ends the stream with `embedding_unavailable`.
10. **Open (owner).** Published outbox rows are never pruned.
11. **Open (owner).** Request coalescing is per API process only; cross-process coalescing would need a Redis lease.
12. **Open (owner).** Run on the real corpus, extend the golden set to 30 questions, run `make eval` with real models and confirm Phase 1.
