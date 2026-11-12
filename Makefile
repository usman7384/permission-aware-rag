.PHONY: up down migrate seed api worker inference test test-int test-int-context test-int-remote eval bench-load bench-eval lint

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run python -m rag.common.migrate

seed:
	uv run python -m rag.common.seed

worker:
	uv run python -m rag.worker.main

api:
	uv run uvicorn rag.api.main:app --loop asyncio:SelectorEventLoop

inference:
	uv run uvicorn --factory rag.inference.main:app_from_env --host 127.0.0.1 --port 8100

test:
	uv run pytest -m unit

test-int:
	uv run pytest -m integration

test-int-context: export CONTEXTUAL_CHUNKING = true
test-int-context:
	uv run pytest -m integration

# The integration fixtures start a fake-model inference service at this URL for the run.
test-int-remote: export INFERENCE_URL = http://127.0.0.1:8101
test-int-remote:
	uv run pytest -m integration

eval:
	uv run python -m rag.evals.score

bench-load:
	uv run python -m rag.evals.benchmark

bench-eval:
	uv run python -m rag.evals.score --benchmark data/beir/scifact --name scifact

lint:
	uv run ruff check .
	uv run ruff format --check .
