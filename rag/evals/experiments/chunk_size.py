import argparse

from rag.common.config import Settings
from rag.evals.benchmark import read_corpus, read_qrels
from rag.evals.experiments import common


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    corpus, qrels = read_corpus(args.benchmark), read_qrels(args.benchmark, args.split)
    docs = common.sample_corpus(corpus, qrels, args.limit, args.docs)
    rows, metrics = [], {}
    for child in args.children:
        for with_parents in (True, False):
            # Without parents, each parent is one child, so the context holds only children.
            parent = settings.parent_tokens if with_parents else child
            variant = f"c{child}{'p' if with_parents else ''}"
            s = common.variant_settings(settings, variant, child_tokens=child, parent_tokens=parent)
            stats = await common.build_variant(settings, s, docs, args.user)
            m = await common.evaluate_variant(s, args, docs, with_answers=True)
            await common.drop_variant(settings, s)
            keep = ("questions", "recall@10", "ndcg@10", "faithfulness")
            metrics[variant] = {k: m[k] for k in keep} | {"chunks": stats["chunks"]}
            parents = "yes" if with_parents else "no"
            row = [child, parents, stats["chunks"], m["recall@10"], m["ndcg@10"], m["faithfulness"]]
            rows.append(row)
    llm = common.llm_name(settings)
    await common.save_run(settings, args, {"documents": len(docs), "llm": llm, "runs": metrics})
    headers = ["child_tokens", "parents", "chunks", "recall@10", "ndcg@10", "faithfulness"]
    return common.table(headers, rows) + f"\n{len(docs)} documents, answers and judge: {llm} LLM"


def parser() -> argparse.ArgumentParser:
    description = "Recall and faithfulness by child size, with and without parents."
    parser = common.parser(description, "chunk_size")
    parser.set_defaults(limit=100)
    parser.add_argument("--children", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--docs", type=int, default=1000, help="documents in the sample corpus")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
