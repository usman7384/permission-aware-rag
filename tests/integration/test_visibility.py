import httpx
import pytest
from fixture_corpus import CODE_WORDS, READABLE


@pytest.mark.parametrize("api_key", sorted(READABLE))
async def test_code_word_returns_its_document_exactly_when_readable(
    client: httpx.AsyncClient, api_key: str
) -> None:
    for path, word in CODE_WORDS.items():
        response = await client.post(
            "/search", json={"question": word}, headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        paths = {parent["path"] for parent in response.json()["parents"]}
        assert paths <= READABLE[api_key], word
        assert (path in paths) == (path in READABLE[api_key]), word


async def test_unknown_api_key_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/search", json={"question": "ZEBRA-7"}, headers={"X-API-Key": "dev-nobody"}
    )
    assert response.status_code == 401
    assert response.json() == {"code": "unauthorized", "message": "unknown API key"}
