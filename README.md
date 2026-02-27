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

# Query (baseline — summaries only)
uv run python main.py query "What are the most common complaints?" --model gpt-4o-mini

# Query (enhanced — with inline quotes)
uv run python main.py query "What are the most common complaints?" --model gpt-4o-mini --mode enhanced
```

## Benchmark

Runs all 6 queries in both modes and evaluates quality, cost, and latency:

```bash
uv run python benchmark.py
# Results saved to outputs/benchmark_results.json
```

Skip the coherence LLM call (faster, no extra cost):
```bash
uv run python benchmark.py --no-coherence
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
| Avg tokens/query | ~342 | ~1,256 |
| Avg cost/query | ~$0.0019 | ~$0.0055 |
| Avg TTFT | ~38ms | ~31ms |
| Avg total time | ~2.5s | ~5.5s |
| Verbatim accuracy | — | 88% |
| Coherence | — | 4.3 / 5 |

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
│   └── APPROACH.md           # Approach & design decisions
└── tests/                   # Unit tests (pytest)
```

## Tests

```bash
uv run pytest
```
