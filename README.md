# RAG Feedback Search with Inline Quotes

A RAG pipeline over user feedback data that generates answers with **verbatim inline quotes** attributed to their source `feedback_record_id`.

## How It Works

```
User Query → Embed → ChromaDB Search → Retrieved Summaries + Raw Feedback → LLM → Streamed Answer with Quotes
```

Feedback summaries are indexed via `text-embedding-3-small`. At query time, the top-k summaries **and** their raw feedback records are passed to the LLM, which produces a prose answer with block quotes cited by record ID.

```
Users report the app freezes during basic tasks.

> "The app FREEZES, doing SIMPLE things like: selecting something, copying and pasting." — f8c057fb-0d61-5dab-a809-3ae0add2f4e5
```

## Two Modes

| Mode | Context passed to LLM | Output |
|---|---|---|
| `baseline` | Summaries only | Plain prose answer |
| `enhanced` | Summaries + raw verbatim feedback | Prose with inline block quotes cited by `feedback_record_id` |

The retriever, indexer, and streaming pipeline are identical in both modes — only the prompt changes.

## Setup

```bash
# Install dependencies
uv sync

# Add your OpenAI API key
cp .env.example .env

# Extract data
unzip data.zip
```

## Usage

```bash
# Build the vector index
uv run python main.py index --data-dir data --index-path .chroma_db

# Query — baseline (summaries only)
uv run python main.py query "What are the most common complaints?" --model gpt-4o-mini

# Query — enhanced (with inline quotes)
uv run python main.py query "What are the most common complaints?" --model gpt-4o-mini --mode enhanced
```

## Benchmark

Runs all 6 queries in both modes and evaluates quality, cost, and latency. Defaults to GPT-4o:

```bash
uv run python benchmark.py
# Results saved to outputs/benchmark_results.json
```

Skip the coherence LLM call (faster, no extra cost):
```bash
uv run python benchmark.py --no-coherence
```

### Benchmark Output Format

Each entry in `benchmark_results.json` represents one query in one mode:

```json
{
  "query": "What are the most common complaints about app performance?",
  "version": "baseline | enhanced",
  "output": "<full LLM answer text>",
  "input_tokens": 239,
  "output_tokens": 124,
  "total_tokens": 363,
  "cost_usd": 0.001838,
  "ttft_s": 0.027,
  "total_time_s": 1.763,

  // enhanced mode only — null for baseline
  "num_quotes": 5,
  "verbatim_rate": 1.0,
  "citation_rate": 1.0,
  "hallucination_rate": 0.0,
  "coherence_score": 5.0,
  "coherence_reasoning": "<LLM judge explanation>",
  "quotes_detail": [
    {
      "extracted_quote": "<text as written by LLM>",
      "feedback_record_id": "<uuid cited by LLM>",
      "actual_feedback_content": "<ground truth raw feedback>",
      "verbatim_match": true,
      "citation_correct": true,
      "hallucinated": false
    }
  ]
}
```

## Evaluation Dimensions

| Dimension | Method | Cost |
|---|---|---|
| Verbatim accuracy | Substring match (programmatic) | Free |
| Citation correctness | Record ID lookup (programmatic) | Free |
| Answer coherence | LLM-as-judge via GPT-4o | ~$0.002/query |

## Baseline vs Enhanced

| Metric | Baseline | Enhanced |
|---|---|---|
| Avg tokens/query | ~337 | ~1,177 |
| Avg cost/query | ~$0.0019 | ~$0.0059 |
| Avg TTFT | ~37ms | ~82ms |
| Avg total time | ~2.2s | ~5.7s |
| Verbatim accuracy | — | 93% |
| Coherence | — | 4.3 / 5 |

Model: GPT-4o. Enhanced mode is ~3.2× more expensive and ~2.5× slower due to raw feedback content added to the prompt (~650 extra input tokens per query).

## Project Structure

```
├── main.py                  # CLI entry point
├── benchmark.py             # Baseline vs enhanced evaluation runner
├── src/
│   ├── indexer.py           # Embeds summaries into ChromaDB
│   ├── retriever.py         # Semantic search
│   ├── llm_client.py        # Prompt builder + streaming generation
│   ├── rag_pipeline.py      # Orchestrator
│   ├── evaluator.py         # Verbatim, citation, coherence checks
│   ├── eval_parser.py       # Block quote regex parser
│   └── cost_tracker.py      # Token count + latency tracking
├── outputs/
│   ├── benchmark_results.json
│   └── APPROACH.md          # Approach & design decisions
└── tests/                   # Unit tests (pytest)
```

## Tests

```bash
uv run pytest
```