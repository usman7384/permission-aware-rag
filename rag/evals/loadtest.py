"""Locust user that `experiments/loadtest.py` runs: asks benchmark questions through `POST /ask`."""

import itertools
import json
import uuid
from pathlib import Path

from locust import HttpUser, constant, events, task
from locust.argument_parser import LocustArgumentParser

# Shared by every simulated user, so each question is asked once before any repeats.
NEXT = itertools.count()


@events.init_command_line_parser.add_listener
def add_arguments(parser: LocustArgumentParser) -> None:
    parser.add_argument("--questions", required=True, help="JSON file of question strings")
    parser.add_argument("--api-key", required=True, help="X-API-Key of the benchmark user")
    parser.add_argument("--session-prefix", required=True, help="start of every session ID")


class Asker(HttpUser):
    wait_time = constant(0)

    def on_start(self) -> None:
        options = self.environment.parsed_options
        self.questions = json.loads(Path(options.questions).read_text(encoding="utf-8"))
        self.headers = {"X-API-Key": options.api_key}
        self.prefix = options.session_prefix

    @task
    def ask(self) -> None:
        """One question in a new session, so it is never rewritten; fails without `done`."""
        body = {
            "session_id": f"{self.prefix}{uuid.uuid4()}",
            "question": self.questions[next(NEXT) % len(self.questions)],
        }
        with self.client.post(
            "/ask", json=body, headers=self.headers, name="ask", catch_response=True
        ) as response:
            if "event: done" not in response.text:
                response.failure("the stream did not end with a done event")
