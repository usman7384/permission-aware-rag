import argparse
import dataclasses
import uuid
from dataclasses import dataclass

from rag.api import pipeline
from rag.api.generate import ANSWER_SYSTEM, assemble
from rag.api.rewrite import REWRITE_SYSTEM, rewrite_prompt
from rag.common.config import Settings
from rag.evals.experiments import common
from rag.evals.score import Golden, mean

# Stands in for the earlier answer in a follow-up's history.
PREVIOUS_ANSWER = "Based on the documents, the answer is in [1]."


@dataclass(frozen=True)
class Prices:
    """Dollars per million tokens."""

    input: float
    cached_input: float
    output: float
    small_input: float
    small_output: float
    min_cached_tokens: int


@dataclass(frozen=True)
class Tokens:
    """Mean tokens per query of the rewrite (small model) and answer (large model) calls."""

    rewrite_in: float
    rewrite_out: float
    answer_in: float
    answer_system: float
    answer_out: float


def usd_per_1k(
    t: Tokens, p: Prices, followup_share: float, hit_rate: float, prompt_cache: bool
) -> dict[str, float]:
    """Dollars per 1,000 queries by stage. Follow-ups are rewritten before the cache lookup;
    a cache hit skips the answer; with prompt caching the system prompt is billed at the
    cached price if it is long enough for the provider to cache."""
    cached = t.answer_system if prompt_cache and t.answer_system >= p.min_cached_tokens else 0.0
    rewrite = followup_share * (t.rewrite_in * p.small_input + t.rewrite_out * p.small_output)
    answer = (t.answer_in - cached) * p.input + cached * p.cached_input + t.answer_out * p.output
    answer *= 1 - hit_rate
    return {"rewrite": rewrite / 1000, "answer": answer / 1000, "total": (rewrite + answer) / 1000}


async def answer_tokens(
    deps: pipeline.Deps, llm: common.CountingLLM, question: Golden
) -> tuple[float, float]:
    """Prompt and completion tokens of answering one question; zero if it abstains."""
    async with deps.pg.connection() as conn:
        principals, _ = await pipeline.expand(conn, question.user)
    vector = await pipeline.embed(deps.embedder, question.question)
    request_id = str(uuid.uuid4())
    parents, degraded = await pipeline.search(
        deps, principals, question.question, vector, request_id, {}
    )
    s, count = deps.settings, deps.embedder.count_tokens
    context = assemble(parents, "rerank" not in degraded, s, count, question.question)
    if context is None:
        return 0.0, 0.0
    before = (llm.prompt_tokens, llm.completion_tokens)
    await llm.complete(
        ANSWER_SYSTEM, context.user, model=s.llm_model or "", max_tokens=s.answer_max_tokens
    )
    return llm.prompt_tokens - before[0], llm.completion_tokens - before[1]


async def rewrite_tokens(
    deps: pipeline.Deps, llm: common.CountingLLM, questions: list[Golden]
) -> tuple[float, float]:
    """Mean tokens of rewriting each question as a follow-up to the one before it."""
    for previous, question in zip(questions, questions[1:], strict=False):
        turns = [("user", previous.question), ("assistant", PREVIOUS_ANSWER)]
        prompt = rewrite_prompt(turns, question.question)
        s = deps.settings
        model, max_tokens = s.llm_small_model or "", s.rewrite_max_tokens
        await llm.complete(REWRITE_SYSTEM, prompt, model=model, max_tokens=max_tokens)
    calls = max(llm.calls, 1)
    return llm.prompt_tokens / calls, llm.completion_tokens / calls


async def measure(
    deps: pipeline.Deps, questions: list[Golden], top_ks: list[int]
) -> dict[int, Tokens]:
    """Mean tokens per query at each RERANK_KEEP."""
    inner = common.fake_llm_unless_configured(deps.settings) or deps.llm
    count = deps.embedder.count_tokens
    rewrite = await rewrite_tokens(deps, common.CountingLLM(inner, count), questions)
    system = count(ANSWER_SYSTEM)
    tokens = {}
    for k in top_ks:
        at_k = dataclasses.replace(
            deps, settings=deps.settings.model_copy(update={"rerank_keep": k})
        )
        llm = common.CountingLLM(inner, count)
        pairs = [await answer_tokens(at_k, llm, q) for q in questions]
        answered = sum(1 for prompt, _ in pairs if prompt) / len(pairs)
        prompt_in, prompt_out = mean([p for p, _ in pairs]), mean([c for _, c in pairs])
        tokens[k] = Tokens(*rewrite, prompt_in, system * answered, prompt_out)
    return tokens


async def experiment(settings: Settings, args: argparse.Namespace) -> str:
    prices = Prices(
        args.input_price,
        args.cached_input_price,
        args.output_price,
        args.small_input_price,
        args.small_output_price,
        args.min_cached_tokens,
    )
    deps = await pipeline.open_deps(settings)
    try:
        questions = await common.labelled_questions(deps.pg, args)
        tokens = await measure(deps, questions, args.top_k)
    finally:
        await pipeline.close_deps(deps)
    rows, runs = [], {}
    for k, t in tokens.items():
        share, hits = args.followup_share, args.hit_rate
        variants = {
            "plain": usd_per_1k(t, prices, share, 0.0, False),
            "prompt_cache": usd_per_1k(t, prices, share, 0.0, True),
            "answer_cache": usd_per_1k(t, prices, share, hits, False),
            "both": usd_per_1k(t, prices, share, hits, True),
        }
        runs[str(k)] = {"tokens": dataclasses.asdict(t), "usd_per_1k": variants}
        plain = variants["plain"]
        rows.append(
            [k, t.rewrite_in, t.answer_in, t.answer_out, plain["rewrite"], plain["answer"]]
            + [v["total"] for v in variants.values()]
        )
    llm_used = common.llm_name(settings)
    metrics = {
        "questions": len(questions),
        "llm": llm_used,
        "prices": dataclasses.asdict(prices),
        "top_k": runs,
    }
    await common.save_run(settings, args, metrics)
    headers = ["top_k", "rewrite_in", "answer_in", "answer_out", "$rewrite", "$answer"]
    headers += ["$total", "$prompt_cache", "$answer_cache", "$both"]
    footer = (
        f"\n$ per 1,000 queries; {len(questions)} queries, {llm_used} LLM, follow-up share "
        f"{args.followup_share}, answer-cache hit rate {args.hit_rate}, system prompt "
        f"{tokens[args.top_k[0]].answer_system:.0f} tokens (cached from {args.min_cached_tokens})"
    )
    return common.table(headers, rows) + footer


def parser() -> argparse.ArgumentParser:
    parser = common.parser("Tokens and cost per query by stage.", "cost")
    parser.set_defaults(limit=20)
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10], help="RERANK_KEEP")
    parser.add_argument("--input-price", type=float, default=3.0, help="$/M input tokens")
    parser.add_argument("--cached-input-price", type=float, default=0.3, help="$/M cached input")
    parser.add_argument("--output-price", type=float, default=15.0, help="$/M output tokens")
    parser.add_argument("--small-input-price", type=float, default=1.0, help="small model input")
    parser.add_argument("--small-output-price", type=float, default=5.0, help="small model out")
    parser.add_argument(
        "--min-cached-tokens", type=int, default=1024, help="shortest prefix a provider caches"
    )
    parser.add_argument("--followup-share", type=float, default=0.3, help="share rewritten")
    parser.add_argument("--hit-rate", type=float, default=0.3, help="answer-cache hit rate")
    return parser


if __name__ == "__main__":
    common.run_main(experiment, parser().parse_args())
