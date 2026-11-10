import asyncio
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        kind = Path(item.path).relative_to(TESTS_DIR).parts[0]
        item.add_marker(getattr(pytest.mark, kind))


def pytest_asyncio_loop_factories(config, item):
    # psycopg's async mode cannot run on the Windows proactor loop.
    return {"selector": asyncio.SelectorEventLoop}
