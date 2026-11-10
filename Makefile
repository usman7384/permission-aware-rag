.PHONY: up down migrate seed api test test-int lint

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run python -m rag.common.migrate

seed:
	uv run python -m rag.common.seed

api:
	uv run uvicorn rag.api.main:app --loop asyncio:SelectorEventLoop

test:
	uv run pytest -m unit

test-int:
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .
