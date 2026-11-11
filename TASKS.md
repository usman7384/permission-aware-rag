# TASKS.md

Build order for the laptop RAG system. Work top to bottom. A milestone is finished when every test it names passes, the whole suite is green and `make lint` is clean. Tick the boxes as you go.

Test names are file names under `tests/unit/` or `tests/integration/`. Each bullet under "Tests" is at least one test function.

## Phase 1

### M0. Scaffold and stores

Spec: sections 2, 3, 5, 6, 10 (`/healthz`), 12.

- [x] Repository layout, `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`, empty `DECISIONS.md`.
- [x] `docker-compose.yml` with the four stores, pinned tags, volumes and the settings in spec section 2.
- [x] `migrations/001_init.sql` and `make migrate`.
- [x] `rag/common/`: `config.py`, the four store clients, `ids.py`, `text.py`, `keys.py`, `log.py`, and `models.py` with the three interfaces and their fakes.
- [x] `GET /healthz`.
- [x] The twelve fixture files and `make seed`.

Tests:

- `unit/test_ids.py`: IDs are deterministic and differ by path, version and ordinal.
- `unit/test_text.py`: `normalize_query` and `extract_entities` on at least six cases each, including punctuation, case and numbers.
- `unit/test_keys.py`: the fingerprint ignores principal order; the cache key changes with query, fingerprint and index version.
- `unit/test_fakes.py`: the fake embedder is deterministic and gives higher similarity to texts sharing words.
- `integration/test_health.py`: all checks true with the stores up.

Owner action: none.

### M1. Ingestion

Spec: sections 9.1, 9.2, 6.

- [x] Watcher with start-up reconciliation.
- [x] Worker consumer loop, reclaim loop and dead-letter handling.
- [x] Markdown parser and chunker.
- [x] Index writes in the order of spec section 9.2, steps 6 to 9.
- [x] Delete jobs.

Tests:

- `unit/test_parse.py`: headings split sections; a fenced code block and a table stay whole.
- `unit/test_chunk.py`: no parent exceeds `PARENT_TOKENS`, no child exceeds `CHILD_TOKENS`, children concatenate back to the parent text.
- `integration/test_ingest.py`: ingesting the fixture corpus yields 12 documents, and the same chunk count in Qdrant and OpenSearch.
- `integration/test_ingest.py`: re-ingesting an unchanged file writes nothing.
- `integration/test_ingest.py`: changing a file bumps its version and leaves no chunk of the old version in any store.
- `integration/test_ingest.py`: deleting a file removes its chunks and parents and sets `deleted_at`.
- `integration/test_dead_letter.py`: a file that makes the parser raise ends in `ingest:dead` after `MAX_DELIVERIES` attempts and is acknowledged.

Covers success check 2.

### M2. Permissions and retrieval

Spec: sections 7, 8 (steps 1, 2, 6, 8 to 12).

- [x] `expand_principals`, `document_readers`, `readable_docs`.
- [x] Stamping `principals` during ingestion.
- [x] `authenticate` (without the rate limit), `expand`, `embed`, `retrieve`, `rrf`, `recheck`, `rerank`, `hydrate` (Postgres only for now).
- [x] A temporary `POST /search` endpoint returning the parents, used by tests; remove it in M3.

Tests:

- `unit/test_rrf.py`: scores match the formula on a hand-worked example; ties break by `chunk_id`; a chunk in both lists outranks one in a single list at the same rank.
- `integration/test_permissions.py`: `expand_principals` for ana, ben and cal matches spec section 12, including the nested `group:staff`.
- `integration/test_permissions.py`: `document_readers` for `specs/api-limits.md` contains `group:eng` and `user:ben`.
- `integration/test_default_deny.py`: cal gets zero results for every code word. This test stays in the suite permanently.
- `integration/test_visibility.py`: searching each of the twelve code words as ana, ben and cal returns a document exactly when spec section 12 says the user can read it.
- `integration/test_retrieve.py`: a code-word query finds its document through the sparse branch alone.
- `integration/test_retrieve.py`: with one branch forced to time out, results come from the other and `degraded` names the failed branch.
- `integration/test_recheck.py`: a chunk whose version is behind `documents.version` is dropped.

Covers success check 4.

### M3. Generation

Spec: sections 8 (steps 13 to 16 without cache writes), 8.2, 8.3, 10 (`/ask`).

- [x] `prompts/answer_v1.txt`.
- [x] `assemble`, `generate`, `output_guard`, `persist` (turns and the request row).
- [x] `POST /ask` with the SSE events of spec section 10.
- [x] A minimal CLI client, `python -m rag.api.client "question" --key dev-ana`.

Tests:

- `unit/test_assemble.py`: best-at-both-ends order for 1, 2, 3 and 5 parents; the lowest-scored parents are dropped to fit the budget.
- `unit/test_output_guard.py`: an out-of-range citation marker is removed; an email not in the context is redacted, one in the context is kept; an answer with no valid citation yields the `uncited` warning.
- `integration/test_ask.py`: a fixture question as ana streams `token` events, then `citations`, then `done`, and the cited path is the expected file.
- `integration/test_ask.py`: asking for a bare code word from a document ana cannot read abstains, with no LLM call.
- `integration/test_ask.py`: a bad API key returns 401 as plain JSON.

Covers success check 3.

Owner action: add `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` and `LLM_SMALL_MODEL` to `.env` to try real answers. Tests do not need them.

### M4. Permission changes

Spec: sections 7.2, 9.3, 10 (`/admin/tuples`).

- [x] `POST /admin/tuples`.
- [x] Invalidator consumer with the `tuple_changed` actions (re-stamp only; cache purge arrives in M5).

Tests:

- `integration/test_tuple_changes.py`: removing `group:eng` viewer `folder:specs` makes ana's next query exclude specs documents before the invalidator has run.
- `integration/test_tuple_changes.py`: after the invalidator runs, the `principals` payload of those chunks no longer contains `group:eng` in Qdrant or OpenSearch.
- `integration/test_tuple_changes.py`: adding `user:cal` viewer one document makes it retrievable by cal after the re-stamp, and no other document.
- `integration/test_tuple_changes.py`: removing ana from `group:eng` changes her fingerprint and results with no re-stamp.
- `integration/test_tuple_changes.py`: a non-admin key gets 401 from `/admin/tuples`.

Covers success check 5.

### M5. Caches and invalidation

Spec: sections 8 (steps 5, 7, 12, 16), 8.4, 9.3, 6.5.

- [x] Exact cache, semantic cache, hydration cache.
- [x] Cache writes after the response.
- [x] Invalidator purge for `doc_changed`, `doc_deleted` and re-stamped documents.

Tests:

- `integration/test_cache.py`: the second identical question returns `cache_tier: exact` and makes no retrieval or LLM call.
- `integration/test_cache.py`: a question differing only in case and trailing punctuation hits the exact cache.
- `integration/test_cache.py`: with the threshold lowered for the test, a reworded question hits the semantic cache; the same rewording with a different number in it does not.
- `integration/test_cache.py`: ana's cached answer is never served to ben for the same question.
- `integration/test_cache.py`: editing a document in the answer's context evicts the `ans:` key, its `answer_cache` point and its `chunk:` keys.
- `integration/test_cache.py`: an abstention and a degraded answer are not cached.
- `integration/test_cache.py`: an expired `ans:` key behind a semantic hit is treated as a miss and its point is deleted.

Covers success check 6.

### M6. Sessions, guardrails and rate limit

Spec: sections 8 (steps 1, 3, 4), 8.1, 10 (`/feedback`).

- [x] `prompts/rewrite_v1.txt` and `rewrite`.
- [x] `input_guard`.
- [x] Rate limit in `authenticate`.
- [x] `POST /feedback`.

Tests:

- `unit/test_input_guard.py`: each rejection code on at least two inputs, and three benign questions that pass.
- `integration/test_sessions.py`: a second question in a session is rewritten using the first; a first question is not sent to the rewriter.
- `integration/test_sessions.py`: when the rewriter fails, the original question is used and `degraded` contains `rewrite`.
- `integration/test_rate_limit.py`: request number `RATE_LIMIT_PER_MIN + 1` in one minute returns 429.
- `integration/test_feedback.py`: feedback is stored; an unknown `request_id` returns 404.

### M7. Evaluation, tracing and fallbacks

Spec: sections 11, 13, 10 (`/admin/faults`).

- [x] Stage timings recorded for all sixteen steps.
- [x] Fault wrappers and `POST /admin/faults`.
- [x] Extractive fallback.
- [x] `rag/evals/golden.jsonl` with the twelve fixture questions, `score.py`, `prompts/judge_v1.txt`, `make eval`.

Tests:

- `integration/test_timings.py`: a cache-miss request stores a timing for every step name; a cache hit stores only the steps that ran.
- `integration/test_faults.py`: one test per row of the table in spec section 11.
- `integration/test_faults.py`: setting a fault back to `ok` restores normal answers.
- `integration/test_eval.py`: with the fake models, `score.py` writes an `eval_runs` row with recall@10, MRR, faithfulness and per-stage latency, and recall@10 is 1.0 on the fixture questions.

Covers success checks 7 and 8.

### M8. Parsing, context and packaging

Spec: sections 9.2 (steps 2 and 4), 2.

- [x] PDF parsing behind the same `parse` function; `.pdf` added to the watcher.
- [x] `prompts/context_v1.txt` and `contextualize`, switched by `CONTEXTUAL_CHUNKING`.
- [x] Dockerfiles for the API and the worker, added to `docker-compose.yml` under a profile so that `make up` still starts only the stores.
- [x] `README.md`: what it is, the architecture in a few lines, prerequisites, how to run from a fresh clone, how to run tests and the evaluation, pinned image tags, and an empty results table for the owner's measured numbers.

Tests:

- `integration/test_pdf.py`: a small fixture PDF with one heading and one table is ingested, and its table text is retrievable.
- `integration/test_context.py`: with contextual chunking on, each indexed child text starts with its parent's context and `parents.context` is filled; with it off, neither happens.
- All earlier tests still pass with contextual chunking on.

Covers success check 1 end to end. Phase 1 ends here. Stop and report to the owner.

Owner action: run the system on the real corpus, extend the golden set to 30 questions, run `make eval` with real models, fill the results table, and confirm Phase 1.

## Phase 2

Do not start until the owner confirms Phase 1. Each experiment is one script in `rag/evals/experiments/` that reads its settings from arguments, writes one `eval_runs` row and prints one table. No experiment changes pipeline behaviour for normal requests.

### M9. Scale-up

- [x] A loader for a public retrieval benchmark with relevance labels (the owner chooses the dataset and size), writing through the normal index functions with batching.
- [x] Vector quantization and on-disk original vectors in Qdrant, behind settings.
- [x] A generator for synthetic folders, groups and tuples with a target filter selectivity.
- [x] `score.py` gains nDCG@10 and recall@100 against the benchmark labels.

### M10. Experiments

One checkbox per script. The question in brackets is what the result must let the owner answer.

- [x] `ablation.py`: nDCG@10 and recall@100 for sparse only, dense only, hybrid, hybrid plus rerank. [Why hybrid? Why rerank?]
- [x] `scale_memory.py`: memory, build time, p95 latency and recall at two corpus sizes for float32, int8 and binary quantization. [How does this scale?]
- [x] `filtered_search.py`: recall and latency as filter selectivity goes from 50% to 0.1%, with and without an exact-scan fallback. [What if a user can see very little?]
- [ ] `loadtest.py`: p50, p95, p99 and throughput per stage at rising concurrency, retrieval measured separately from generation. [Where does it break first?]
- [ ] `cache_behaviour.py`: hit rate and false-hit rate of the semantic cache across thresholds on a skewed query log. [Why this threshold?]
- [ ] `freshness.py`: seconds from file change to searchable and to cache eviction. [How fresh is the index?]
- [ ] `failure_injection.py`: answer quality and latency under each fault. [What happens when a dependency is down?]
- [x] `model_migration.py`: build a second index under a new `INDEX_VERSION`, compare on the labels, switch with no failed requests. [How do you upgrade the embedding model?]
- [x] `chunk_size.py`: recall and faithfulness for 64, 128 and 256-token children, with and without parents. [Why these chunk sizes?]
- [x] `contextual_chunking.py`: recall with and without chunk context on a sample, and the indexing cost added. [Is the cost worth it?]
- [ ] `cost.py`: tokens and cost per query by stage, and the effect of the caches, prompt caching and top-k. [How do you cut cost?]

Owner action: write the results table and the design document from these runs.
