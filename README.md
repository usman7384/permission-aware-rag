# rag-laptop

A RAG service for one laptop: API, ingestion worker and evaluation over Postgres, Redis, Qdrant and OpenSearch. `SPEC.md` is the design; `DECISIONS.md` lists the choices that still matter.

Needs Docker, [uv](https://docs.astral.sh/uv/) and GNU make (Windows: `winget install ezwinports.make`, run from Git Bash). Images: `postgres:18.6`, `redis:8.10.3` (host port 6380), `qdrant/qdrant:v1.20.0`, `opensearchproject/opensearch:3.9.0`, base `python:3.12.16-slim-trixie`.

## Setup and run

```
cp .env.example .env                       # add LLM_* settings for real answers
uv sync
make up && make migrate && make seed       # stores, schema, fixture users and keys
cp -r tests/fixtures/corpus/* corpus/      # optional fixture documents
make worker                                # terminal 1
make api                                   # terminal 2: http://127.0.0.1:8000
uv run python -m rag.api.client "How many paid leave days do we get?" --key dev-ben
make inference                             # optional: models on :8100, then set INFERENCE_URL
docker compose --profile app up -d         # optional: API, worker and inference in containers
```

## Test and eval

```
make test               # unit
make test-int           # integration (make up first)
make test-int-context   # integration with CONTEXTUAL_CHUNKING=true
make test-int-remote    # integration through a fake-model inference service on :8101
make lint
make eval               # score rag/evals/golden.jsonl and store an eval_runs row
```
