# rag-laptop

A retrieval-augmented generation service that runs on one laptop. See `SPEC.md` for the design and `TASKS.md` for build progress. The full guide arrives with milestone 8.

## Quick start

Prerequisites: Docker, [uv](https://docs.astral.sh/uv/), GNU make.

```
cp .env.example .env
make up        # start Postgres, Redis, Qdrant, OpenSearch
make migrate   # Postgres schema, Qdrant collections, OpenSearch index
make seed      # fixture users, groups, tuples and API keys
make api       # http://localhost:8000/healthz
make worker    # index CORPUS_DIR and watch it for changes
make test      # unit tests
make test-int  # integration tests (needs make up)
make lint
```

## Pinned images

| Store | Image |
| --- | --- |
| Postgres | `postgres:18.6` |
| Redis | `redis:8.10.3` (host port 6380) |
| Qdrant | `qdrant/qdrant:v1.20.0` |
| OpenSearch | `opensearchproject/opensearch:3.9.0` |
