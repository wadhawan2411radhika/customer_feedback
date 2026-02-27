#!/usr/bin/env python3
"""
benchmark.py — Runs each query in baseline and enhanced mode.

Per-record output:
  query, version, output, input_tokens, output_tokens, total_tokens,
  cost_usd, ttft_s, total_time_s,
  verbatim_rate, citation_rate, hallucination_rate, coherence_score,
  quotes_detail (enhanced only)

Usage:
    uv run python benchmark.py
    uv run python benchmark.py --no-coherence
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.evaluator import evaluate_query, load_feedback_records
from src.llm_client import OpenAIClient
from src.rag_pipeline import RAGPipeline
from src.retriever import FeedbackRetriever

load_dotenv()

DEFAULT_QUERIES = [
    "What are the most common complaints about app performance?",
    "Summarize what users love most about Canva",
    "What usability issues are users facing?",
    "What feature requests or suggestions are users making?",
    "How do users feel about Canva's pricing and paid features?",
    "What are users saying about the recent update?",
]


async def run(
    pipeline: RAGPipeline,
    query: str,
    mode: str,
    feedback_records: dict,
    coherence_client: AsyncOpenAI | None,
    coherence_model: str,
) -> dict:
    """Run one query in one mode, return a fully populated result dict."""
    chunks = []
    async for chunk in pipeline.query(query, mode=mode):
        chunks.append(chunk)
    answer = "".join(chunks)

    result = pipeline.llm_client.last_cost.to_dict(output=answer)

    if mode == "enhanced":
        eval_result = await evaluate_query(
            query=query,
            answer=answer,
            feedback_records=feedback_records,
            coherence_client=coherence_client,
            coherence_model=coherence_model,
        )
        quotes_detail = [
            {
                "extracted_quote": v.quote.text,
                "feedback_record_id": v.quote.record_id,
                "actual_feedback_content": feedback_records.get(v.quote.record_id, "NOT FOUND"),
                "verbatim_match": v.verbatim_match,
                "citation_correct": v.citation_correct,
                "hallucinated": v.hallucinated,
            }
            for v in eval_result.verdicts
        ]
        result.update({
            "num_quotes": eval_result.num_quotes,
            "verbatim_rate": eval_result.verbatim_rate,
            "citation_rate": eval_result.citation_rate,
            "hallucination_rate": eval_result.hallucination_rate,
            "coherence_score": eval_result.coherence_score,
            "coherence_reasoning": eval_result.coherence_reasoning,
            "quotes_detail": quotes_detail,
        })
    else:
        result.update({
            "num_quotes": None,
            "verbatim_rate": None,
            "citation_rate": None,
            "hallucination_rate": None,
            "coherence_score": None,
            "coherence_reasoning": None,
            "quotes_detail": None,
        })

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs enhanced")
    parser.add_argument("--data-dir",        type=Path, default=Path("data"))
    parser.add_argument("--index-path",      type=Path, default=Path(".chroma_db"))
    parser.add_argument("--model",           type=str,  default="gpt-4o")
    parser.add_argument("--coherence-model", type=str,  default="gpt-4o")
    parser.add_argument("--top-k",           type=int,  default=5)
    parser.add_argument("--no-coherence",    action="store_true")
    parser.add_argument("--output-json",     type=Path, default=Path("outputs/benchmark_results.json"))
    args = parser.parse_args()

    print(f"Loading feedback records from {args.data_dir}...")
    feedback_records = load_feedback_records(args.data_dir)
    print(f"Loaded {len(feedback_records)} records.\n")

    pipeline = RAGPipeline(
        retriever=FeedbackRetriever(index_path=args.index_path, top_k=args.top_k),
        llm_client=OpenAIClient(model=args.model),
    )
    coherence_client = None if args.no_coherence else AsyncOpenAI()

    results = []
    total = len(DEFAULT_QUERIES) * 2

    for i, query in enumerate(DEFAULT_QUERIES):
        for j, mode in enumerate(("baseline", "enhanced")):
            n = i * 2 + j + 1
            print(f"[{n}/{total}] mode={mode}  query={query[:55]!r}")

            result = await run(
                pipeline=pipeline,
                query=query,
                mode=mode,
                feedback_records=feedback_records,
                coherence_client=coherence_client,
                coherence_model=args.coherence_model,
            )
            results.append(result)

            print(f"  → tokens={result['total_tokens']}  cost=${result['cost_usd']}  "
                  f"ttft={result['ttft_s']}s  total={result['total_time_s']}s"
                  + (f"  verbatim={result['verbatim_rate']:.0%}  "
                     f"coherence={result['coherence_score']}/5"
                     if mode == "enhanced" and result["verbatim_rate"] is not None else ""))

            await asyncio.sleep(0.5)

    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())