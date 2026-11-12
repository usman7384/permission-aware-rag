# rag-laptop

A retrieval-augmented generation (RAG) service that runs on one laptop. It answers questions over a folder of documents (Markdown, text and PDF), enforces per-user permissions, serves repeated questions from cache, degrades gracefully when a dependency fails, and measures its own quality and latency. No LLM or RAG framework is used: fusion, caching, permission checks and prompt assembly are written by hand.

`SPEC.md` is the full design, `TASKS.md` the build order, `DECISIONS.md` the choices made where the spec was silent.

## Architecture

- **Worker** (`rag/worker`): watches `CORPUS_DIR`, queues one job per changed file in a Redis stream, parses (Markdown splitter, docling for PDF), chunks into parents and children, optionally adds an LLM-written context per parent, stamps each chunk with the principals allowed to read it, and writes Qdrant (dense vectors), OpenSearch (BM25) and Postgres (parents, versions). It also runs the outbox relay, which publishes change events committed to Postgres to a Redis stream, and the invalidator, which re-stamps permissions and purges caches on those events.
- **API** (`rag/api`): `POST /ask` runs sixteen timed steps: authenticate and rate-limit, expand the user's groups, input guard, rewrite the question from the session, exact cache, embed, semantic cache (hits checked against live versions and permissions), hybrid retrieval with a permission filter (an exact scan when the user can read few chunks), reciprocal rank fusion, a live permission recheck, cross-encoder reranking, parent hydration, context assembly, streamed generation, output guard (citations, PII), persist and cache writes. Answers stream as server-sent events.
- **Stores**: Postgres is the source of truth (documents, parents, permission tuples, requests, evaluations); Qdrant, OpenSearch and Redis (job streams, caches, rate limits, fault flags) can be rebuilt from it and the files.
- **Evaluation** (`rag/evals/score.py`): recall@10, MRR, LLM-judged faithfulness and per-stage latency over `rag/evals/golden.jsonl`, stored in `eval_runs`.

## Prerequisites

- Docker with Compose v2.24 or later (Docker Desktop on Windows and macOS).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for the project if needed.
- GNU make.
  - Linux and macOS: already installed, or from the system package manager (`xcode-select --install` on macOS).
  - Windows: `winget install ezwinports.make`, then run every command below from Git Bash. If `make` is not found, add its folder to `PATH`, for example `export PATH="$LOCALAPPDATA/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin:$PATH"`.
- About 3 GB of disk for the Python environment and, on first use, the models: the embedder and reranker (about 200 MB) and the docling layout and table models (about 500 MB, downloaded the first time a PDF is parsed).
- Free host ports 5432 (Postgres), 6380 (Redis), 6333 and 6334 (Qdrant), 9200 (OpenSearch), 8000 (API) and 8100 (inference service, optional).

## Run from a fresh clone

```
git clone <repository-url> rag-laptop && cd rag-laptop
cp .env.example .env
uv sync                                   # Python 3.12 environment from uv.lock
make up                                   # start Postgres, Redis, Qdrant, OpenSearch
make migrate                              # Postgres schema, Qdrant collections, OpenSearch index
make seed                                 # fixture users, groups, tuples and API keys (spec section 12)
cp -r tests/fixtures/corpus/* corpus/     # optional: the twelve fixture documents
make worker                               # terminal 1: index corpus/, then watch it
make api                                  # terminal 2: http://127.0.0.1:8000
curl http://127.0.0.1:8000/healthz
uv run python -m rag.api.client "How many paid leave days do we get?" --key dev-ben
```

`make up` starts only the four stores. The fixture API keys are `dev-ana`, `dev-ben`, `dev-cal` and `dev-admin` (admin); they are for local development only.

Real answers need an OpenAI-compatible endpoint in `.env`: `LLM_BASE_URL` (ending before `/chat/completions`, for example `https://api.openai.com/v1`), `LLM_API_KEY`, `LLM_MODEL` and `LLM_SMALL_MODEL`. Without them, `/healthz` reports `llm: false` and `/ask` returns the extractive fallback with `llm` in `degraded`. To try everything without models or keys, set `MODEL_BACKEND=fake` in `.env`.

Contextual chunking (an LLM-written context prepended to every chunk before indexing) is off by default. Set `CONTEXTUAL_CHUNKING=true` in `.env`, then change `INDEX_VERSION` (and run `make migrate`) so the index is rebuilt.

### Redis on host port 6380

Compose publishes Redis on host port **6380**, not 6379, so it cannot collide with another Redis already running on the laptop. To use a different port, change the left side of `"6380:6379"` under `redis` in `docker-compose.yml` and the port in `REDIS_URL` in `.env` (for example `REDIS_URL=redis://127.0.0.1:6390/0`). Store addresses use `127.0.0.1` rather than `localhost` because some systems resolve `localhost` to IPv6 first.

### API and worker in containers

The API and the worker can also run in containers, from one image built by `Dockerfile`. They are in the Compose profile `app`, so `make up` never starts them.

```
docker compose --profile app build
docker compose --profile app up -d       # stores, API on http://127.0.0.1:8000, worker
docker compose --profile app down
```

The containers read `.env` but always reach the stores by their Compose service names, watch `./corpus` through a bind mount (with polling, because file events from a host folder do not reach a container on every platform) and keep downloaded models in the `model-cache` volume. Run `make migrate` and `make seed` from the host first. Do not run `make worker` on the host and the worker container at the same time.

### Inference service (optional)

The embedder and the reranker can run in their own process, so model work cannot slow the API's other requests and overload is refused instead of queued:

```
make inference                            # terminal 3: models on http://127.0.0.1:8100
INFERENCE_URL=http://127.0.0.1:8100 make api      # and the same for make worker
```

With `INFERENCE_URL` unset (the default) the models run inside the API and the worker as before. The service runs at most `INFERENCE_CONCURRENCY` model calls at once with `INFERENCE_QUEUE_MAX` more waiting, and answers 503 at once beyond that; the API then answers in fused order with `rerank` in `degraded` (or, if the query cannot be embedded, an `embedding_unavailable` error event). In Compose, the `app` profile starts the `inference` service and points the API and worker containers at it; its port is published on 127.0.0.1 only, because it has no API key.

## Tests and evaluation

```
make test               # unit tests: no services, no network, under 30 seconds
make test-int           # integration tests against the running stores (make up first)
make test-int-context   # the same integration tests with CONTEXTUAL_CHUNKING=true
make test-int-remote    # the same integration tests through a fake-model inference service on port 8101
make lint               # ruff check and ruff format --check
make eval               # score the golden set, print the table, store an eval_runs row
```

Integration tests use the Postgres database `test_rag`, Qdrant collections and an OpenSearch index prefixed `test_`, and Redis database 15, so they never touch development data. They use the deterministic fake models and never call a paid API. The first PDF test downloads the docling models once. `make eval` uses the models and stores configured in `.env`; for numbers that mean anything, use `MODEL_BACKEND=real`, real LLM keys and your own corpus and golden questions.

### Benchmark (Phase 2)

```
make bench-load         # download BEIR SciFact into data/ and index it into separate stores
make bench-eval         # recall@10, recall@100, nDCG@10 and MRR against the SciFact test labels
```

`make bench-load` runs `python -m rag.evals.benchmark`; it downloads the dataset once from the BEIR site into the git-ignored `data/beir/`, and writes into the Postgres database `rag_bench` and Qdrant collections and an OpenSearch index prefixed `bench_`, never into development data. Every document gets a synthetic folder, and the user `user:bench` gets group memberships that let it read a chosen share of them: `--folders 1000 --selectivity 0.01` gives 1% (`--help` lists the other options, such as `--limit` for a smaller load and `--dataset` for another BEIR dataset). Changing only `--selectivity` re-runs quickly, because it changes memberships, not the index. A different `--folders` needs a fresh `--database` and `--prefix`. `QDRANT_QUANTIZATION` (`int8` or `binary`) and `QDRANT_VECTORS_ON_DISK=true` choose how the chunk vectors are stored when a collection is created.

### Experiments (Phase 2)

Each experiment is one script in `rag/evals/experiments/`, run against the loaded benchmark (`make bench-load` first). Each reads its sizes from arguments (`--help` lists them), writes one `eval_runs` row in `rag_bench` and prints one table:

```
uv run python -m rag.evals.experiments.ablation             # sparse, dense, hybrid, hybrid + rerank
uv run python -m rag.evals.experiments.scale_memory         # float32, int8, binary at two sizes
uv run python -m rag.evals.experiments.filtered_search      # selectivity 50% to 0.1%, HNSW vs the planned search
uv run python -m rag.evals.experiments.model_migration      # second embedder under INDEX_VERSION v2
uv run python -m rag.evals.experiments.chunk_size           # 64/128/256-token children, with/without parents
uv run python -m rag.evals.experiments.contextual_chunking  # with and without chunk context
uv run python -m rag.evals.experiments.loadtest             # per-stage latency at 1, 2, 4, 8 users (Locust)
                                                            # --hits: plus one user repeating a cached question
uv run python -m rag.evals.experiments.cache_behaviour      # cache hit and false-hit rates by threshold
uv run python -m rag.evals.experiments.freshness            # file change to searchable and to eviction
uv run python -m rag.evals.experiments.failure_injection    # quality and latency under each fault
uv run python -m rag.evals.experiments.cost                 # tokens and $ per query by stage and top-k
```

`scale_memory` and `filtered_search` copy the benchmark's vectors into scratch Qdrant collections and delete them afterwards. `chunk_size` and `contextual_chunking` index a sample of the corpus into their own databases (`rag_bench_<variant>`) and indexes, deleted after scoring. `model_migration` leaves `bench_chunks_v2` in place. `loadtest` serves the API in its own process on `--port` (default 8765) and drives it with Locust (`rag/evals/loadtest.py`); run it with `INFERENCE_URL` set and `make inference` running to also see the share of answers degraded by rerank and the service's 503 rate. `freshness` runs the real worker on a temporary folder with its own database and empties the scratch Redis database `--redis-db` (default 1). `failure_injection` sets `fault:` keys in the configured Redis, so do not run it next to an API you are using; it clears them when it ends. `cost` takes prices in dollars per million tokens as arguments. Without `LLM_*` settings, answers, judgements and contexts come from the fake LLM, and the table says so.

`tests/fixtures/make_pdf.py` regenerates the fixture PDF byte for byte (`uv run python tests/fixtures/make_pdf.py`).

## Pinned images

| Service | Image |
| --- | --- |
| Postgres | `postgres:18.6` |
| Redis | `redis:8.10.3` (host port 6380) |
| Qdrant | `qdrant/qdrant:v1.20.0` |
| OpenSearch | `opensearchproject/opensearch:3.9.0` |
| API, worker and inference base | `python:3.12.16-slim-trixie`, with `uv` from `ghcr.io/astral-sh/uv:0.9.13` |

## Results

Measured by the owner with `make eval` on the real corpus and real models. Empty until then.

| Date | Corpus | Questions | Models | recall@10 | MRR | Faithfulness | p50 latency (ms) | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |
