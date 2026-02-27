"""
Three evaluation dimensions for the inline quotes RAG system.

1. Verbatim accuracy  — programmatic substring check
2. Citation accuracy  — programmatic record lookup
3. Answer coherence   — LLM-as-judge (GPT-4o)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from src.eval_parser import ParsedQuote, parse_quotes


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuoteVerdict:
    quote: ParsedQuote
    verbatim_match: bool
    citation_correct: bool
    found_in_other: Optional[str] = None
    hallucinated: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.quote.text,
            "record_id": self.quote.record_id,
            "verbatim_match": self.verbatim_match,
            "citation_correct": self.citation_correct,
            "hallucinated": self.hallucinated,
            "found_in_other": self.found_in_other,
        }


@dataclass
class QueryEvalResult:
    query: str
    answer: str
    num_quotes: int
    verdicts: list[QuoteVerdict] = field(default_factory=list)
    verbatim_rate: float = 0.0
    citation_rate: float = 0.0
    hallucination_rate: float = 0.0
    coherence_score: Optional[float] = None
    coherence_reasoning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "num_quotes": self.num_quotes,
            "verbatim_rate": self.verbatim_rate,
            "citation_rate": self.citation_rate,
            "hallucination_rate": self.hallucination_rate,
            "coherence_score": self.coherence_score,
            "coherence_reasoning": self.coherence_reasoning,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------------------
# Feedback record loader
# ---------------------------------------------------------------------------

def load_feedback_records(data_dir: Path) -> dict[str, str]:
    """Returns {record_id: verbatim_content} for all feedback_record.json files."""
    records = {}
    for summary_dir in data_dir.iterdir():
        if not summary_dir.is_dir():
            continue
        record_file = summary_dir / "feedback_record.json"
        if not record_file.exists():
            continue
        with open(record_file) as f:
            data = json.load(f)
        try:
            record_id = data["id"]
            content = data["attributes"]["content"]["string"]["values"][0]
            if record_id and content:
                records[record_id] = content
        except (KeyError, IndexError):
            continue
    return records


# ---------------------------------------------------------------------------
# Dimension 1 & 2 — Verbatim + Citation (programmatic)
# ---------------------------------------------------------------------------

def evaluate_verbatim_and_citation(
    quotes: list[ParsedQuote],
    feedback_records: dict[str, str],
) -> list[QuoteVerdict]:
    verdicts = []
    for quote in quotes:
        source_content = feedback_records.get(quote.record_id, "")
        if quote.text.lower() in source_content.lower():
            verdicts.append(QuoteVerdict(quote=quote, verbatim_match=True, citation_correct=True))
            continue

        # Not in claimed record — check others to distinguish wrong attribution vs hallucination
        found_in = next(
            (rid for rid, content in feedback_records.items()
             if rid != quote.record_id and quote.text.lower() in content.lower()),
            None,
        )
        verdicts.append(QuoteVerdict(
            quote=quote,
            verbatim_match=False,
            citation_correct=False,
            found_in_other=found_in,
            hallucinated=(found_in is None),
        ))
    return verdicts


def compute_rates(verdicts: list[QuoteVerdict]) -> tuple[float, float, float]:
    if not verdicts:
        return 0.0, 0.0, 0.0
    n = len(verdicts)
    return (
        sum(v.verbatim_match for v in verdicts) / n,
        sum(v.citation_correct for v in verdicts) / n,
        sum(v.hallucinated for v in verdicts) / n,
    )


# ---------------------------------------------------------------------------
# Dimension 3 — Coherence (LLM-as-judge)
# ---------------------------------------------------------------------------

COHERENCE_PROMPT = """You are evaluating an AI-generated answer that includes verbatim quotes from user feedback.

Rate the answer 1–5 for COHERENCE — how naturally quotes are woven into the prose.

- 5: Every quote directly supports the claim before it; prose flows naturally
- 4: Most quotes well-placed; minor awkwardness in one or two spots
- 3: Some quotes feel forced or loosely connected to surrounding prose
- 2: Quotes mostly disconnected from claims; hard to follow
- 1: Quotes dumped in with no connection to prose; incoherent

Query: {query}

Answer:
{answer}

Respond in this exact format:
SCORE: <1-5>
REASONING: <one or two sentences>"""


def _parse_coherence_response(raw: str) -> tuple[float, str]:
    score, reasoning = 0.0, ""
    for line in raw.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return score, reasoning


async def evaluate_coherence(
    query: str,
    answer: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
) -> tuple[float, str]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": COHERENCE_PROMPT.format(query=query, answer=answer)}],
        temperature=0.0,
    )
    return _parse_coherence_response(response.choices[0].message.content.strip())


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

async def evaluate_query(
    query: str,
    answer: str,
    feedback_records: dict[str, str],
    coherence_client: Optional[AsyncOpenAI] = None,
    coherence_model: str = "gpt-4o",
) -> QueryEvalResult:
    quotes = parse_quotes(answer)
    result = QueryEvalResult(query=query, answer=answer, num_quotes=len(quotes))

    if quotes:
        result.verdicts = evaluate_verbatim_and_citation(quotes, feedback_records)
        result.verbatim_rate, result.citation_rate, result.hallucination_rate = compute_rates(result.verdicts)

    if coherence_client:
        result.coherence_score, result.coherence_reasoning = await evaluate_coherence(
            query=query, answer=answer, client=coherence_client, model=coherence_model,
        )

    return result