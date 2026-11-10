FROM ghcr.io/astral-sh/uv:0.9.13 AS uv

FROM python:3.12.16-slim-trixie
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY rag rag
COPY prompts prompts
COPY migrations migrations
RUN uv sync --locked --no-dev
