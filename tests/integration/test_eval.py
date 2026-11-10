from rag.api.pipeline import Deps
from rag.evals.score import GOLDEN, evaluate, load_golden


async def test_score_stores_an_eval_run_with_full_recall_on_the_fixtures(api_deps: Deps) -> None:
    questions = load_golden(GOLDEN)
    assert len(questions) == 12
    run_id, metrics = await evaluate(api_deps, questions, "test")
    async with api_deps.pg.connection() as conn:
        cur = await conn.execute(
            "SELECT name, config, metrics FROM eval_runs WHERE run_id = %s", (run_id,)
        )
        name, config, stored = await cur.fetchone()
    assert name == "test" and stored == metrics
    assert config["model_backend"] == "fake" and "llm_api_key" not in config
    assert stored["recall@10"] == 1.0
    assert 0 < stored["mrr"] <= 1
    assert stored["faithfulness"] == 1.0
    for step in ("expand", "embed", "retrieve", "rerank", "generate", "output_guard"):
        assert set(stored["latency_ms"][step]) == {"p50", "p95"}
