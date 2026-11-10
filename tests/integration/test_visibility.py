import pytest
from conftest import search_paths
from fixture_corpus import CODE_WORDS, READABLE

from rag.api.pipeline import Deps


@pytest.mark.parametrize("api_key", sorted(READABLE))
async def test_code_word_returns_its_document_exactly_when_readable(
    api_deps: Deps, api_key: str
) -> None:
    for path, word in CODE_WORDS.items():
        paths = await search_paths(api_deps, api_key, word)
        assert paths <= READABLE[api_key], word
        assert (path in paths) == (path in READABLE[api_key]), word
