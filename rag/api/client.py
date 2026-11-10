import argparse
import json
import sys
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from rag.common.config import Settings


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """(event, data) pairs from the lines of a `text/event-stream` body."""
    name, data = "message", ""
    for line in lines:
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data += line.removeprefix("data:").strip()
        elif not line and data:
            yield name, json.loads(data)
            name, data = "message", ""


def show(name: str, data: dict[str, Any]) -> None:
    if name == "token":
        print(data["text"], end="", flush=True)
    elif name == "citations":
        print()
        for item in data["items"]:
            print(f"[{item['n']}] {item['path']}")
    elif name == "done":
        print(json.dumps(data), file=sys.stderr)
    else:
        print(f"{name}: {json.dumps(data)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG service a question.")
    parser.add_argument("question")
    parser.add_argument("--key", required=True, help="API key, for example dev-ana")
    parser.add_argument("--session", default=str(uuid.uuid4()), help="session id")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    args = parser.parse_args()
    settings = Settings()
    timeout = httpx.Timeout(settings.store_timeout_s, read=settings.llm_timeout_s)
    body = {"session_id": args.session, "question": args.question}
    with httpx.stream(
        "POST", f"{args.url}/ask", json=body, headers={"X-API-Key": args.key}, timeout=timeout
    ) as response:
        if response.status_code != 200:
            print(response.read().decode(), file=sys.stderr)
            sys.exit(1)
        for name, data in parse_sse(response.iter_lines()):
            show(name, data)


if __name__ == "__main__":
    main()
