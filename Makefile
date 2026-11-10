.PHONY: up down migrate seed api worker test test-int test-int-context eval lint

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

test:
	uv run pytest -m unit

test-int:
	uv run pytest -m integration

test-int-context: export CONTEXTUAL_CHUNKING = true
test-int-context:
	uv run pytest -m integration

eval:
	uv run python -m rag.evals.score

lint:
	uv run ruff check .
	uv run ruff format --check .
