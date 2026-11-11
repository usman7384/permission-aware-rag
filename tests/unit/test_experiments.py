from rag.common.config import Settings
from rag.common.models import FakeLLM
from rag.common.text import tokens
from rag.evals.benchmark import BenchDoc
from rag.evals.experiments.ablation import variant_ms
from rag.evals.experiments.common import overlap, sample_corpus, table, variant_settings
from rag.evals.experiments.contextual_chunking import CountingLLM
from rag.evals.experiments.filtered_search import readable_groups, with_folder_principals
from rag.evals.experiments.scale_memory import ram_mb
from rag.evals.synthetic import synthetic_acl


def test_table_aligns_columns_and_rounds_floats() -> None:
    text = table(["name", "value"], [["a", 0.12345], ["long name", 2]])
    assert text.splitlines() == ["     name  value", "        a  0.123", "long name      2"]


def test_sample_keeps_relevant_documents_of_the_first_queries_then_fills_in_corpus_order() -> None:
    docs = [BenchDoc(str(i), "", "") for i in range(1, 9)]
    qrels = {"q0": {"9": 0}, "q1": {"7": 1, "2": 0}, "q2": {"5": 2}, "q3": {"8": 1}}
    assert [d.id for d in sample_corpus(docs, qrels, 2, 4)] == ["1", "2", "5", "7"]
    assert [d.id for d in sample_corpus(docs, qrels, 2, 1)] == ["5", "7"]
    assert [d.id for d in sample_corpus(docs, qrels, None, 3)] == ["5", "7", "8"]
    missing = qrels | {"q4": {"99": 1}}
    assert [d.id for d in sample_corpus(docs, missing, None, 4)] == ["1", "5", "7", "8"]


def test_variant_settings_get_their_own_database_and_index_version() -> None:
    base = Settings(pg_dsn="postgresql://rag:rag@127.0.0.1:5432/rag_bench", store_prefix="bench_")
    variant = variant_settings(base, "c64p", child_tokens=64)
    assert "dbname=rag_bench_c64p" in variant.pg_dsn
    assert variant.chunks_name == "bench_chunks_c64p" and variant.child_tokens == 64
    assert base.chunks_name == "bench_chunks_v1"


def test_overlap_is_the_share_of_true_neighbours_found() -> None:
    assert overlap(["a", "b", "x"], ["a", "b", "c", "d"]) == 0.5
    assert overlap([], []) == 1.0


def test_ram_follows_bytes_per_dimension() -> None:
    assert ram_mb(1024, 256, "none") == 1.0
    assert ram_mb(1024, 256, "int8") == 0.25
    assert ram_mb(1024, 256, "binary") == 1 / 32


def test_hybrid_waits_for_the_slower_branch_and_rerank_adds_on() -> None:
    ms = variant_ms({"embed": 5.0, "dense": 10.0, "sparse": 12.0, "rerank": 100.0})
    assert ms == {"sparse": 12.0, "dense": 15.0, "hybrid": 15.0, "hybrid+rerank": 115.0}


def test_points_get_their_folder_group_and_the_user_reads_the_first_folders() -> None:
    points = [({"doc_id": d, "principals": ["old"]}, [0.0]) for d in ("b", "a", "c", "a")]
    stamped = with_folder_principals(points, 2, "user:bench")
    assert [p["principals"] for p, _ in stamped] == [
        ["group:filtered-f0001"],
        ["group:filtered-f0000"],
        ["group:filtered-f0000"],
        ["group:filtered-f0000"],
    ]
    acl = synthetic_acl("filtered", 3, 2, 0.5, "user:bench")
    assert readable_groups(acl, "user:bench") == ["group:filtered-f0000"]


async def test_counting_llm_counts_calls_and_tokens() -> None:
    llm = CountingLLM(FakeLLM(), lambda text: len(tokens(text)))
    reply = await llm.complete("KIND: context", "one two", model="", max_tokens=8)
    assert reply == "This chunk is from the document."
    assert (llm.calls, llm.prompt_tokens, llm.completion_tokens) == (1, 4, 6)
