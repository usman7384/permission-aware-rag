import httpx
from fixture_corpus import CODE_WORDS


async def test_cal_gets_zero_results_for_every_code_word(client: httpx.AsyncClient) -> None:
    for word in CODE_WORDS.values():
        response = await client.post(
            "/search", json={"question": word}, headers={"X-API-Key": "dev-cal"}
        )
        assert response.status_code == 200
        assert response.json()["parents"] == [], word
        assert response.json()["degraded"] == []
