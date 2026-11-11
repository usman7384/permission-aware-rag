import argparse
from typing import Any

from qdrant_client import AsyncQdrantClient

from rag.common import qdrant_
from rag.common.config import Settings
from rag.evals.experiments import common
from rag.evals.score import mean, percentile

QUANTIZATIONS = ("none", "int8", "binary")
BYTES_PER_DIMENSION = {"none": 4.0, "int8": 1.0, "binary": 1 / 8}


def ram_mb(points: int, dim: int, quantization: str) -> float:
    """Megabytes of vectors Qdrant keeps in RAM to search: the float32 originals, or only the
    quantized copy when the originals are on disk."""
    return points * dim * BYTES_PER_DIMENSION[quantization] / 2**20


async def measure(
    client: AsyncQdrantClient,
    settings: Settings,
    points: common.Points,
    vectors: list[list[float]],
    principals: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a collection of the points with the settings' quantization, search it, drop it."""
    quantization = settings.qdrant_quantization
    build_s = await common.build_collection(client, settings, points, args.threshold_kb)
    search = (client, settings, vectors, principals, args.top_k)
    truth, _ = await common.timed_searches(*search, exact=True)
    found, ms = await common.timed_searches(*search, exact=False)
    await common.drop_collections(client, settings)
    return {
        "chunks": len(points),
        "quantization": quantization,
        "ram_mb": ram_mb(len(points), len(points[0][1]), quantization),
        "build_s": build_s,
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        f"recall@{args.top_k}": mean(
            [common.overlap(f, t) for f, t in zip(found, truth, strict=True)]
        ),
    }


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    client = qdrant_.connect(settings)
    runs = []
    try:
        points = await qdrant_.scroll_chunks(client, settings, max(args.sizes))
        vectors = await common.query_vectors(settings, args, args.queries)
        principals = sorted({p for payload, _ in points for p in payload["principals"]})
        for size in args.sizes:
            for quantization in args.quantizations:
                # Quantized vectors stay in RAM and the originals go to disk, as in decision 74.
                variant = {
                    "index_version": f"scale_{quantization}_{min(size, len(points))}",
                    "qdrant_quantization": quantization,
                    "qdrant_vectors_on_disk": quantization != "none",
                }
                s = settings.model_copy(update=variant)
                runs.append(await measure(client, s, points[:size], vectors, principals, args))
    finally:
        await client.close()
    await common.save_run(settings, args, {"queries": len(vectors), "runs": runs})
    return common.table(list(runs[0]), [list(run.values()) for run in runs])


def parser() -> argparse.ArgumentParser:
    parser = common.parser("Memory, build time, p95 latency and recall per quantization.", "scale")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 18099], help="chunks")
    parser.add_argument("--quantizations", nargs="+", default=list(QUANTIZATIONS))
    parser.add_argument("--queries", type=int, default=300, help="benchmark queries to search")
    parser.add_argument("--top-k", type=int, default=10, help="neighbours compared with exact")
    parser.add_argument("--threshold-kb", type=int, default=10, help=common.THRESHOLD_HELP)
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
