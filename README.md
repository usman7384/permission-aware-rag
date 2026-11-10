# rag-laptop

A retrieval-augmented generation (RAG) service that runs on one laptop. It answers questions over a folder of documents (Markdown, text and PDF), enforces per-user permissions, serves repeated questions from cache, degrades gracefully when a dependency fails, and measures its own quality and latency. No LLM or RAG framework is used: fusion, caching, permission checks and prompt assembly are written by hand.

`SPEC.md` is the full design, `TASKS.md` the build order, `DECISIONS.md` the choices made where the spec was silent.

## Architecture

- **Worker** (`rag/worker`): watches `CORPUS_DIR`, queues one job per changed file in a Redis stream, parses (Markdown splitter, docling for PDF), chunks into parents and children, optionally adds an LLM-written context per parent, stamps each chunk with the principals allowed to read it, and writes Qdrant (dense vectors), OpenSearch (BM25) and Postgres (parents, versions). It also runs the invalidator, which re-stamps permissions and purges caches on change events.
- **API** (`rag/api`): `POST /ask` runs sixteen timed steps: authenticate and rate-limit, expand the user's groups, input guard, rewrite the question from the session, exact cache, embed, semantic cache, hybrid retrieval with a permission filter, reciprocal rank fusion, a live permission recheck, cross-encoder reranking, parent hydration, context assembly, streamed generation, output guard (citations, PII), persist and cache writes. Answers stream as server-sent events.
- **Stores**: Postgres is the source of truth (documents, parents, permission tuples, requests, evaluations); Qdrant, OpenSearch and Redis (job streams, caches, rate limits, fault flags) can be rebuilt from it and the files.
- **Evaluation** (`rag/evals/score.py`): recall@10, MRR, LLM-judged faithfulness and per-stage latency over `rag/evals/golden.jsonl`, stored in `eval_runs`.

## Prerequisites

- Docker with Compose v2.24 or later (Docker Desktop on Windows and macOS).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for the project if needed.
- GNU make.
  - Linux and macOS: already installed, or from the system package manager (`xcode-select --install` on macOS).
  - Windows: `winget install ezwinports.make`, then run every command below from Git Bash. If `make` is not found, add its folder to `PATH`, for example `export PATH="$LOCALAPPDATA/Microsoft/WinGet/Packages/ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe/bin:$PATH"`.
- About 3 GB of disk for the Python environment and, on first use, the models: the embedder and reranker (about 200 MB) and the docling layout and table models (about 500 MB, downloaded the first time a PDF is parsed).
- Free host ports 5432 (Postgres), 6380 (Redis), 6333 and 6334 (Qdrant), 9200 (OpenSearch) and 8000 (API).

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

## Tests and evaluation

```
make test               # unit tests: no services, no network, under 30 seconds
make test-int           # integration tests against the running stores (make up first)
make test-int-context   # the same integration tests with CONTEXTUAL_CHUNKING=true
make lint               # ruff check and ruff format --check
make eval               # score the golden set, print the table, store an eval_runs row
```

Integration tests use the Postgres database `test_rag`, Qdrant collections and an OpenSearch index prefixed `test_`, and Redis database 15, so they never touch development data. They use the deterministic fake models and never call a paid API. The first PDF test downloads the docling models once. `make eval` uses the models and stores configured in `.env`; for numbers that mean anything, use `MODEL_BACKEND=real`, real LLM keys and your own corpus and golden questions.

`tests/fixtures/make_pdf.py` regenerates the fixture PDF byte for byte (`uv run python tests/fixtures/make_pdf.py`).

## Pinned images

| Service | Image |
| --- | --- |
| Postgres | `postgres:18.6` |
| Redis | `redis:8.10.3` (host port 6380) |
| Qdrant | `qdrant/qdrant:v1.20.0` |
| OpenSearch | `opensearchproject/opensearch:3.9.0` |
| API and worker base | `python:3.12.16-slim-trixie`, with `uv` from `ghcr.io/astral-sh/uv:0.9.13` |

## Results

Measured by the owner with `make eval` on the real corpus and real models. Empty until then.

| Date | Corpus | Questions | Models | recall@10 | MRR | Faithfulness | p50 latency (ms) | p95 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |
