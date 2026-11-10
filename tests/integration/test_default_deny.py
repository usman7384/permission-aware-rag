from conftest import search_paths
from fixture_corpus import CODE_WORDS

from rag.api.pipeline import Deps


async def test_cal_gets_zero_results_for_every_code_word(api_deps: Deps) -> None:
    for word in CODE_WORDS.values():
        assert await search_paths(api_deps, "dev-cal", word) == set(), word
