import argparse
import asyncio
import math
import random
from collections import Counter
from dataclasses import dataclass

from rag.api.cache import semantic_match
from rag.common.config import Settings
from rag.common.models import make_embedder
from rag.common.text import extract_entities, normalize_query
from rag.evals.benchmark import read_queries
from rag.evals.experiments import common


@dataclass(frozen=True)
class Logged:
    text: str
    intent: int


def lowered(text: str) -> str:
    bare = text.strip().rstrip("?.!")
    return bare[:1].lower() + bare[1:]


def paraphrases(text: str) -> list[str]:
    """Wordings that ask the same thing: as written, lowercased without end punctuation, and
    two rephrasings."""
    return [
        text,
        text.strip().rstrip("?.!").lower(),
        f"Is it true that {lowered(text)}?",
        f"Tell me whether {lowered(text)}.",
    ]


def negated(text: str) -> str:
    """Nearly the same words, the opposite question: a cached answer to `text` is wrong here."""
    return f"It is false that {lowered(text)}."


def query_log(
    texts: list[str], requests: int, skew: float, negated_share: float, rng: random.Random
) -> list[Logged]:
    """Requests drawn from a Zipf law over the texts' intents; each intent is one text's
    paraphrases, and each text also has a rarer negated intent."""
    intents = [paraphrases(t) for t in texts] + [[negated(t)] for t in texts]
    weights = [1 / rank**skew for rank in range(1, len(texts) + 1)]
    weights += [negated_share * w for w in weights]
    picks = rng.choices(range(len(intents)), weights, k=requests)
    return [Logged(rng.choice(intents[i]), i) for i in picks]


def replay(
    log: list[Logged], similarity: dict[tuple[str, str], float], threshold: float
) -> dict[str, float]:
    """Run the log through the exact and semantic tiers of one user; every miss is answered
    and cached. Returns hit, false-hit and missed-reuse rates over all requests."""
    exact: dict[str, int] = {}
    cached: list[Logged] = []
    cached_intents: set[int] = set()
    counts: Counter[str] = Counter()
    for query in log:
        served = exact.get(normalize_query(query.text))
        tier = "exact"
        if served is None and cached:
            nearest = max(cached, key=lambda c: similarity[c.text, query.text])
            score = similarity[nearest.text, query.text]
            if semantic_match(score, extract_entities(nearest.text), query.text, threshold):
                served, tier = nearest.intent, "semantic"
        if served is None:
            counts["missed_reuse"] += query.intent in cached_intents
            exact[normalize_query(query.text)] = query.intent
            cached.append(query)
            cached_intents.add(query.intent)
            continue
        counts[tier] += 1
        counts["false_hits"] += served != query.intent
    n = len(log)
    rates = {name: counts[name] / n for name in ("exact", "semantic", "false_hits")}
    rates["missed_reuse"] = counts["missed_reuse"] / n
    return {"hit_rate": rates["exact"] + rates["semantic"], **rates}


def similarities(texts: list[str], vectors: list[list[float]]) -> dict[tuple[str, str], float]:
    """Cosine similarity of every ordered pair of the L2-normalized vectors."""
    return {
        (a, b): math.sumprod(u, v)
        for a, u in zip(texts, vectors, strict=True)
        for b, v in zip(texts, vectors, strict=True)
    }


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    texts = list(read_queries(args.benchmark).values())[: args.intents]
    log = query_log(texts, args.requests, args.skew, args.negated_share, random.Random(args.seed))
    unique = sorted({q.text for q in log})
    embedder = make_embedder(settings)
    vectors = await asyncio.to_thread(lambda: [embedder.embed_query(t) for t in unique])
    similarity = similarities(unique, vectors)
    runs = {str(t): replay(log, similarity, t) for t in args.thresholds}
    metrics = {"requests": len(log), "distinct_texts": len(unique), "thresholds": runs}
    await common.save_run(settings, args, metrics)
    headers = ["threshold", "hit_rate", "exact", "semantic", "false_hits", "missed_reuse"]
    rows = [[t, *r.values()] for t, r in runs.items()]
    footer = f"\n{len(log)} requests, {len(unique)} distinct texts, {len(texts) * 2} intents"
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    description = "Hit rate and false-hit rate of the answer caches on a skewed query log."
    parser = common.parser(description, "cache_behaviour")
    parser.add_argument("--intents", type=int, default=200, help="benchmark queries used")
    parser.add_argument("--requests", type=int, default=2000, help="requests in the log")
    parser.add_argument("--skew", type=float, default=1.0, help="Zipf exponent of popularity")
    parser.add_argument("--negated-share", type=float, default=0.1, help="weight of negations")
    parser.add_argument("--seed", type=int, default=7, help="random seed of the log")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.85, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99],
        help="semantic cache thresholds",
    )
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
