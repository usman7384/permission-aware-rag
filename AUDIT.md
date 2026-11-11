# AUDIT.md: hand-built parts vs free-tier tools

Date: 2026-11-11. Free-tier limits were checked on official pricing and docs pages that day. "Unconfirmed" means the page did not show the figure. Measured numbers come from the M10 experiments on BEIR SciFact (5,183 docs, 18,099 chunks) on this laptop. Experiments that needed an LLM used the fake one.

## Summary

| # | Component | Ours | Best free option | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Vector search | Qdrant (self-hosted) | Qdrant Cloud Free | **Keep** |
| 2 | Keyword search | OpenSearch | Qdrant sparse/BM25, or Postgres + ParadeDB `pg_search` | **Consider merging**: one fewer store to run and sync |
| 3 | Embeddings | local bge-small | Voyage (200M free tokens) | **Keep** (free, no quota) |
| 4 | Reranker | local MiniLM cross-encoder | local bge-reranker-v2-m3, or Voyage rerank-2.5 | **Swap or remove**: measured as harmful and the bottleneck |
| 5 | LLM | OpenAI-compatible adapter | Groq free, Gemini free, OpenRouter `:free` | **Use a hosted one** (the adapter already fits) |
| 6 | PDF parsing | docling | already off-the-shelf | **Keep** |
| 7 | Permissions (ReBAC) | tuples + recursive SQL | OpenFGA (Apache-2.0, CNCF) | **Keep for now**; OpenFGA when the model grows |
| 8 | Answer caches | Redis + Qdrant, hand-written | RedisVL SemanticCache | **Keep**, but fix the threshold |
| 9 | Tracing | JSON logs + `timings` JSONB | Langfuse (MIT self-host / Hobby) or Arize Phoenix | **Swap**: the highest-value change |
| 10 | Evaluation | hand-written IR metrics + judge | Ragas / DeepEval | **Keep IR metrics, swap the judge** |
| 11 | Guardrails | regex | Presidio + Llama Prompt Guard 2 | **Add in front of the regex** |
| 12 | Job queue | Redis Streams, hand-written | Dramatiq or arq | **Optional swap** |
| 13 | Load testing | locust | k6 | **Keep** |
| 14 | Hosting | Docker on laptop | Qdrant Cloud, Neon, Upstash free | Demo only; see limits |

## The changes worth making, in order

### 1. Tracing → Langfuse or Phoenix (component 9)
**Ours:** `rag/common/log.py` (11 lines) plus a `timings` JSONB per request. You can measure a step but not see one request's full path, prompt or answer side by side.

**Free options:**
- Langfuse self-hosted (MIT) or its Hobby plan: 50k units/month, 30 days of data, 2 users.
- Arize Phoenix self-hosted (ELv2), or Arize AX free: 25k spans/month.
- LangSmith Developer: 5k traces/month.

**How it plugs in:** wrap each of the sixteen step functions in a span. The `timed()` context manager in `rag/api/pipeline.py` is already the one place to do it. Keep the `timings` column, because `score.py` reads it.

**Why it pays off:** trace UIs, prompt version diffs and links from traces to eval runs take weeks to build, and every serious RAG team runs one of these.

### 2. Reranker: measure a better one, or drop it (component 4)
The measurements argue against the current model:
- **Quality (`ablation.py`):** nDCG@10 is 0.699 for hybrid but 0.680 for hybrid + rerank. Reranking made SciFact worse.
- **Latency:** it adds about 1.6 s per query (p50 ~1,595 ms vs 39 ms).
- **Load (`loadtest.py`):** throughput is capped at ~1.35 req/s however many users there are. The rerank stage alone hits 4.8 s p50 at 8 users, so it is the first thing to break.

**Options:**
- `BAAI/bge-reranker-v2-m3` (MIT, local): a stronger drop-in behind the same `Reranker` protocol.
- Voyage rerank-2.5 / 2.5-lite: 200M free tokens per account.
- Cohere's trial (10 req/min, 1,000 calls/month) and Pinecone's bundled reranker (500 requests/month) are too small to rely on.

**Next step:** add the model as `RERANK_MODEL`, re-run `ablation.py`, and keep reranking only if nDCG beats 0.699. MS MARCO-trained MiniLM is a poor fit for scientific claims, so this may be a domain mismatch. Re-check on your own corpus.

### 3. Guardrails → Presidio + Prompt Guard 2 (component 11)
**Ours:** `rag/api/guard.py` plus four regex patterns. These catch "ignore previous instructions" but not paraphrases, other languages or encoded payloads. The PII regex sees only emails and card-like numbers, not names, phone numbers or IBANs.

**Options:**
- Microsoft Presidio (MIT): NER-based PII detection and redaction, local.
- Llama Prompt Guard 2 86M: an injection classifier, local, Llama community licence.
- Lakera Guard has a free tier, but its limits are unconfirmed.

**How it plugs in:** keep the regex as the cheap first pass, then call the classifier and Presidio inside `input_guard` and `output_guard`. The 400 codes stay the same.

### 4. Merge keyword search into one store (component 2)
**Ours:** Qdrant and OpenSearch each hold every chunk and its `principals`, and Postgres holds the truth. That is three copies, which is why `recheck`, versioned chunks and the invalidator re-stamp of two stores exist.

**What the ablation says:** sparse alone scores nDCG 0.621. Fusing it with dense moves recall@100 from 0.938 to 0.950 and leaves nDCG@10 flat (0.706 vs 0.699). Keyword search still matters for exact codes (`ZEBRA-7`), so keep it but host it with the vectors.

**Options:**
- **Qdrant sparse vectors (BM25):** dense and sparse prefetch in one Query API call, with RRF fused server-side. OpenSearch goes away, along with `opensearch_.py` (121 lines) and one container with a 512 MB JVM heap.
- **Postgres + pgvector + ParadeDB `pg_search` (AGPL-3.0):** everything, including the permission tuples, sits in one database and one transaction. The recheck query could join directly.

**Trade-off:** you lose the "hybrid across two engines" interview story, and server-side fusion replaces the hand-written `rrf`. Keep the hand-written `rrf` for the learning value, or keep it in experiments only.

### 5. Faithfulness judge → Ragas or DeepEval (component 10)
- Keep `recall@10`, `MRR` and `nDCG@10` in `rag/evals/score.py`. They are a few lines each and fully trustworthy.
- Swap `prompts/judge_v1.txt` for Ragas `faithfulness` or DeepEval `FaithfulnessMetric`. Both are Apache-2.0, need an LLM to judge (the hosted free tiers work), and give numbers people outside this project recognise.

## Leave as they are

- **Vector search (1):** `qdrant_.py` is a thin client, so there is nothing to replace. Qdrant Cloud Free (0.5 vCPU, 1 GB RAM, 4 GB disk) holds SciFact easily: 26.5 MB of float32 vectors, 6.6 MB at int8.
- **Embeddings (3):** local bge-small is free, unthrottled and private. A hosted model is worth it only if an eval shows it wins. Note `model_migration.py`: MiniLM-L6 tied bge-small (nDCG 0.727 vs 0.726).
- **Parsing (6):** docling is MIT and already off-the-shelf. Use LlamaParse (10K credits/month) or Unstructured (10K pages to start) only for scanned or table-heavy PDFs that docling misreads.
- **Permissions (7):** `permissions.py` is 41 lines plus two recursive queries, and it is the core interview story. OpenFGA (used by Okta/Auth0, Docker, Grafana, Supabase) becomes worth it once you need conditions, sharing links, audit trails or millions of tuples. The swap is clean: `expand_principals` and `readable_docs` become `ListObjects` / `BatchCheck` calls, and the recheck gate stays. Permit.io Community (1,000 MAU, 2,000 tuples) is too small for real data.
- **Caching (8):** `cache.py` is 75 lines, about the same size as a RedisVL integration. The tool isn't the problem; the threshold is (see below).
- **Load testing (13):** locust is MIT and Python. k6 is only better if you already use Grafana.

## Optional

- **Job queue (12):** `worker/main.py` (165 lines) plus the Streams helpers in `redis_.py` do consumer groups, reclaim after 30 s and dead-lettering by hand.
  - Dramatiq (LGPL-3.0, Redis broker, retries and a dead-letter queue built in) or arq (MIT, asyncio) would remove most of that.
  - The idempotent versioned writes, which are the part that matters, stay either way.
  - Temporal is the heavyweight answer for multi-step pipelines. It is free to self-host, but Cloud has only 90-day credits.
  - It's worth it if the worker grows. Today the hand-written version is tested and small.
- **LLM (5):** the adapter already speaks the OpenAI protocol, so only `.env` changes.
  - **Groq free:** e.g. gpt-oss-120b at 30 RPM, 1K requests/day.
  - **Gemini Flash / Flash-Lite:** free, but free-tier data is used to improve Google's products. Don't send private documents.
  - **OpenRouter `:free`:** 20 RPM, 50/day, or 1,000/day after a one-off $10 credit purchase.
  - Cerebras is a 30-day trial only.
- **Hosting (14):** Qdrant Cloud Free, Neon Free (1 GB per project, pgvector, scales to zero) and Pinecone Starter (2 GB) are fine for a demo.
  - Upstash Redis Free (500K commands/month) cannot host an always-on Streams worker, because blocking reads consume commands.
  - Supabase Free pauses after a week idle.
  - There is no permanently free managed OpenSearch, which is another reason for change 4.

## Findings from the experiments that are not tool questions

1. **Semantic cache threshold is too loose.** At 0.96, about 9.7% of requests got a wrong cached answer (`cache_behaviour.py`), mostly negated rewordings. At 0.98 it is 3.8%, and the hit rate only drops from 0.876 to 0.851. Raise `SEMANTIC_CACHE_THRESHOLD` to 0.98, or check negation words in `semantic_match`. No library fixes this.
2. **Filtered search collapses without a fallback.** At 1% readable, ANN recall is 0.266, while the exact scan gives 1.0 at 25 ms (`filtered_search.py`). A user who can see little gets bad answers today. The pipeline needs the exact-scan fallback when the readable count is small. This is decision 79, which the owner should approve.
3. **Slow faults don't degrade.** With the defaults, a 5 s slow LLM or reranker stays inside its 30 s / 2 s budget, so the answer is just slow and not marked degraded (decision 58). Either set `FAULT_SLOW_MS` above the timeouts or lower `LLM_TIMEOUT_S`.
4. **Top-k is the cost lever.** Measured with placeholder prices: k=10 costs 2.7× k=3 per query. Prompt caching saves nothing, because the 117-token system prompt is under providers' ~1,024-token minimum.
5. **Freshness is good.** A file change is searchable in 0.32 s p50, and its cache is evicted in 0.41 s.

## Project rules to respect when swapping

`AGENTS.md` allows only the dependencies in `SPEC.md` section 2 unless the reason is recorded in `DECISIONS.md`. It also bans LLM/RAG frameworks (LangChain, LlamaIndex, Haystack). None of the swaps above is a RAG framework, but each new dependency needs a `DECISIONS.md` entry. Ragas has optional LangChain integrations; use its core metrics only.

## Sources

Qdrant pricing https://qdrant.tech/pricing/ · Qdrant hybrid queries https://qdrant.tech/documentation/concepts/hybrid-queries/ · ParadeDB https://www.paradedb.com/docs/deploy/overview · Pinecone https://www.pinecone.io/pricing/ · Weaviate https://weaviate.io/pricing · Voyage https://docs.voyageai.com/docs/pricing · Cohere limits https://docs.cohere.com/docs/rate-limits · Jina https://jina.ai/api-dashboard/rate-limit · FlagEmbedding https://github.com/FlagOpen/FlagEmbedding · Groq https://console.groq.com/docs/rate-limits · Gemini https://aistudio.google.com/docs/pricing · OpenRouter https://openrouter.ai/docs/api-reference/limits · Unstructured https://unstructured.io/pricing · LlamaParse https://www.llamaindex.ai/pricing · OpenFGA https://openfga.dev/ · AuthZed https://authzed.com/pricing · Permit.io https://www.permit.io/pricing · RedisVL https://github.com/redis/redis-vl-python · Langfuse https://langfuse.com/pricing · Arize https://arize.com/pricing/ · LangSmith https://www.langchain.com/pricing · Presidio https://github.com/microsoft/presidio · Prompt Guard 2 https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M · Temporal https://temporal.io/pricing · arq https://github.com/python-arq/arq · Neon https://neon.com/pricing · Upstash https://upstash.com/pricing/redis · Supabase https://supabase.com/pricing · Grafana https://grafana.com/pricing/
