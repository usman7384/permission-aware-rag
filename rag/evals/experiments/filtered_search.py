import argparse
import time
from typing import Any

from qdrant_client import AsyncQdrantClient

from rag.common import qdrant_
from rag.common.config import Settings
from rag.evals.experiments import common
from rag.evals.score import mean, percentile
from rag.evals.synthetic import Acl, folder_group, reader_group, synthetic_acl


def with_folder_principals(points: common.Points, folders: int, user: str) -> common.Points:
    """The points re-stamped so each document's chunks are readable by its synthetic folder's
    group, documents dealt round-robin over the folders in doc_id order."""
    docs = sorted({payload["doc_id"] for payload, _ in points})
    acl = synthetic_acl("filtered", len(docs), folders, 1.0, user)
    group_of = {doc: folder_group(folder) for doc, folder in zip(docs, acl.folders, strict=True)}
    return [({**p, "principals": [group_of[p["doc_id"]]]}, v) for p, v in points]


def readable_groups(acl: Acl, user: str) -> list[str]:
    """The folder groups the user's reader group is a member of."""
    return [obj for subject, _, obj in acl.member_tuples if subject == reader_group(user)]


async def measure(
    client: AsyncQdrantClient,
    settings: Settings,
    vectors: list[list[float]],
    principals: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Recall and latency of the filtered HNSW search, alone and with the exact-scan fallback."""
    start = time.perf_counter()
    readable = await qdrant_.count_chunks(client, settings, principals)
    count_ms = (time.perf_counter() - start) * 1000
    search = (client, settings, vectors, principals, args.top_k)
    truth, exact_ms = await common.timed_searches(*search, exact=True)
    found, ann_ms = await common.timed_searches(*search, exact=False)
    # The fallback counts the readable chunks first, and scans them exactly when they are few.
    use_exact = readable <= args.exact_max_points
    fallback_found = truth if use_exact else found
    fallback_ms = [count_ms + ms for ms in (exact_ms if use_exact else ann_ms)]
    pairs = zip(found, truth, strict=True)
    return {
        "readable_chunks": readable,
        "ann_recall": mean([common.overlap(f, t) for f, t in pairs]),
        "ann_p95_ms": percentile(ann_ms, 95),
        "fallback": "exact" if use_exact else "ann",
        "fallback_recall": mean(
            [common.overlap(f, t) for f, t in zip(fallback_found, truth, strict=True)]
        ),
        "fallback_p95_ms": percentile(fallback_ms, 95),
    }


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    client = qdrant_.connect(settings)
    scratch = settings.model_copy(update={"index_version": "filtered"})
    runs = []
    try:
        points = await qdrant_.scroll_chunks(client, settings, args.max_chunks)
        stamped = with_folder_principals(points, args.folders, args.user)
        build_s = await common.build_collection(client, scratch, stamped, args.threshold_kb)
        vectors = await common.query_vectors(settings, args, args.queries)
        doc_count = len({payload["doc_id"] for payload, _ in points})
        for selectivity in args.selectivities:
            acl = synthetic_acl("filtered", doc_count, args.folders, selectivity, args.user)
            principals = readable_groups(acl, args.user)
            run = await measure(client, scratch, vectors, principals, args)
            runs.append({"selectivity": acl.selectivity, **run})
        await common.drop_collections(client, scratch)
    finally:
        await client.close()
    metrics = {"queries": len(vectors), "chunks": len(points), "build_s": build_s, "runs": runs}
    await common.save_run(settings, args, metrics)
    return common.table(list(runs[0]), [list(run.values()) for run in runs])


def parser() -> argparse.ArgumentParser:
    parser = common.parser("Recall and latency of filtered search by selectivity.", "filtered")
    parser.add_argument("--selectivities", type=float, nargs="+", default=[0.5, 0.1, 0.01, 0.001])
    parser.add_argument("--folders", type=int, default=1000, help="synthetic folders")
    parser.add_argument("--queries", type=int, default=300, help="benchmark queries to search")
    parser.add_argument("--top-k", type=int, default=10, help="neighbours compared with exact")
    parser.add_argument("--threshold-kb", type=int, default=10, help=common.THRESHOLD_HELP)
    parser.add_argument(
        "--exact-max-points", type=int, default=2000, help="fallback: scan exactly up to N chunks"
    )
    parser.add_argument("--max-chunks", type=int, default=1_000_000, help="chunks to copy")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
