import json
import sys
from datetime import UTC, datetime
from typing import Any


def log(event: str, **fields: Any) -> None:
    record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    sys.stdout.flush()
