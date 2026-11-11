import argparse
from collections.abc import AsyncIterator
from dataclasses import dataclass

from rag.common.config import Settings
from rag.common.models import LLM, make_embedder, make_llm
from rag.evals.benchmark import read_corpus, read_qrels
from rag.evals.experiments import common
from rag.worker.chunk import Counter


@dataclass
class CountingLLM:
    """An LLM that counts its calls and their prompt and completion tokens."""

    llm: LLM
    count: Counter
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    async def complete(self, system: str, user: str, *, model: str, max_tokens: int) -> str:
        reply = await self.llm.complete(system, user, model=model, max_tokens=max_tokens)
        self.calls += 1
        self.prompt_tokens += self.count(system) + self.count(user)
        self.completion_tokens += self.count(reply)
        return reply

    def stream(self, system: str, user: str, *, model: str, max_tokens: int) -> AsyncIterator[str]:
        return self.llm.stream(system, user, model=model, max_tokens=max_tokens)


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    corpus, qrels = read_corpus(args.benchmark), read_qrels(args.benchmark, args.split)
    docs = common.sample_corpus(corpus, qrels, args.limit, args.docs)
    # Tokens are counted with the embedding model's tokenizer, as chunk sizes are.
    count = make_embedder(settings).count_tokens
    rows, metrics = [], {}
    for contextual in (False, True):
        arm = "context" if contextual else "plain"
        s = common.variant_settings(settings, arm, contextual_chunking=contextual)
        llm = CountingLLM(common.fake_llm_unless_configured(settings) or make_llm(settings), count)
        stats = await common.build_variant(settings, s, docs, args.user, llm)
        m = await common.evaluate_variant(s, args, docs, with_answers=False)
        await common.drop_variant(settings, s)
        cost = {"llm_calls": llm.calls, "prompt_tokens": llm.prompt_tokens}
        cost |= {"completion_tokens": llm.completion_tokens, "index_s": stats["seconds"]}
        metrics[arm] = {"recall@10": m["recall@10"], "ndcg@10": m["ndcg@10"], **cost}
        rows.append([arm, stats["chunks"], m["recall@10"], m["ndcg@10"], *cost.values()])
    llm_used = common.llm_name(settings)
    await common.save_run(settings, args, {"documents": len(docs), "llm": llm_used, **metrics})
    headers = ["arm", "chunks", "recall@10", "ndcg@10", "llm_calls", "prompt_tok"]
    headers += ["completion_tok", "index_s"]
    footer = f"\n{len(docs)} documents, {m['questions']} queries, contexts: {llm_used} LLM"
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    description = "Recall with and without chunk context, and the indexing cost added."
    parser = common.parser(description, "contextual_chunking")
    parser.set_defaults(limit=100)
    parser.add_argument("--docs", type=int, default=500, help="documents in the sample corpus")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
