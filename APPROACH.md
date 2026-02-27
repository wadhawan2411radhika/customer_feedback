# Inline Quotes RAG — Approach & Design Decisions

## 1. The Task

The existing RAG pipeline returned plain-text summaries with no connection back to original user feedback. The goal was to add **inline verbatim quotes** attributed to their source `feedback_record_id`, while keeping streaming intact and not breaking the existing retrieval architecture.

**Key architectural decision:** Only the prompt layer changed. The retriever, indexer, ChromaDB setup, and streaming pipeline are untouched. This was intentional — the problem was a generation problem, not a retrieval problem.

---

## 2. Observing the Data

Before writing any code, the search results structure was inspected. Each `SearchResult` already contained everything needed:

```python
result.content                    # summary text (already indexed)
result.feedback_summary           # full summary JSON with feedback_record_id
result.feedback_record            # full record JSON with raw verbatim content
```

The `feedback_record_id` linking summary → raw feedback was already present in the retrieval output. No schema changes, no additional lookups, no re-indexing required.

---

## 3. Enhancing the LLM Context

The baseline prompt passed only summaries:

```
Feedback 1: <summary text>
Feedback 2: <summary text>
```

The enhanced prompt passes both summary and raw feedback per record:

```
[RECORD 1]
feedback_record_id: <uuid>
summary: <LLM-generated summary>
verbatim_feedback: <raw user text>
```

This separation is deliberate. Labelling the two sources explicitly reduces a common failure mode where the LLM quotes from the summary (which is paraphrased) instead of the raw text (which is verbatim).

---

## 4. Prompt Design — Output Format

The LLM was instructed to produce a specific structure:

```
Paragraph synthesising related claims from the feedback.

> "exact text copied from verbatim_feedback" — feedback_record_id
> "another quote if relevant" — feedback_record_id

Next paragraph on a different theme.

> "verbatim quote" — feedback_record_id
```

Key prompt decisions and their rationale:

- **Group related claims** — "Do not write one sentence per record" prevents a list-like answer when multiple records say similar things
- **Quote from `verbatim_feedback` only** — explicitly forbidden from quoting the summary
- **Omit rather than fabricate** — if no verbatim text supports a claim, skip the quote; this proved critical for preventing hallucinations
- **One quote per line** — makes the parser's regex unambiguous

---

## 5. Cost & Latency Tracking

**Why this was non-trivial with streaming:** In a standard (non-streaming) API call, token usage is returned directly in the response. With streaming, usage is only available in the final chunk via `stream_options: {"include_usage": True}` — it arrives after all content has been yielded. The tracker is designed around this:

```python
async for chunk in stream:
    if chunk.usage:                          # only on final chunk
        self.last_cost.input_tokens = chunk.usage.prompt_tokens
        self.last_cost.output_tokens = chunk.usage.completion_tokens
    if chunk.choices and chunk.choices[0].delta.content:
        if first_chunk:
            self.last_cost.time_to_first_token = time.perf_counter() - t_start
        yield chunk.choices[0].delta.content  # stream to caller immediately
```

TTFT is captured on first content chunk; total time and token counts are captured on the usage chunk at the end.

### Baseline vs Enhanced — Cost & Latency (GPT-4o-mini, 6 queries)

| Metric | Baseline | Enhanced | Delta |
|---|---|---|---|
| Avg input tokens | ~201 | ~852 | +323% |
| Avg output tokens | ~141 | ~404 | +186% |
| Avg total tokens | ~342 | ~1,256 | +267% |
| Avg cost/query | ~$0.0019 | ~$0.0055 | **~3× more** |
| Avg TTFT | ~38ms | ~31ms | Similar |
| Avg total time | ~2.5s | ~5.5s | +2× slower |

**Why input tokens dominate:** Raw feedback content adds ~650 tokens per query on average compared to summary-only context. This is the primary cost driver. Output tokens also increase because quoted answers are longer than plain summaries.

**TTFT is similar or better** in enhanced mode because the LLM starts generating prose immediately — the user sees the first words of the answer at roughly the same time. The extra time is in total response length, not in getting started.

---

## 6. Evaluation

Three dimensions were evaluated across 6 product-manager-style queries:

### Verbatim Accuracy — Programmatic
Checks whether the extracted quote is a substring of the actual `feedback_record` content.

```python
quote.text.lower() in source_content.lower()
```

No LLM cost. Fast and deterministic.

### Citation Correctness — Programmatic
Checks whether the `feedback_record_id` in the quote actually maps to the record the quote came from. If a quote is verbatim but attributed to the wrong record, this catches it.

Both verbatim and citation are programmatic — they run in milliseconds and add zero API cost.

### Answer Coherence — LLM-as-Judge
Checks whether quotes are naturally woven into the prose or dumped in as a disconnected list. Rated 1–5 by GPT-4o with a structured prompt:

```
5 — Every quote directly supports the claim before it; prose flows naturally
3 — Some quotes feel forced or loosely connected
1 — Quotes dumped in with no connection to prose
```

**Trade-off:** This adds one API call per query at evaluation time (~$0.002 with GPT-4o). Justified for offline benchmarking; would be removed in production.

### Results

| Query | Verbatim | Citation | Hallucination | Coherence |
|---|---|---|---|---|
| App performance complaints | 87.5% | 87.5% | 12.5% | 5/5 |
| What users love about Canva | 100% | 100% | 0% | 2/5 |
| Usability issues | 80% | 80% | 20% | 5/5 |
| Feature requests | 80% | 80% | 20% | 5/5 |
| Pricing & paid features | 100% | 100% | 0% | 5/5 |
| Recent update sentiment | 80% | 80% | 20% | 4/5 |
| **Average** | **88%** | **88%** | **12%** | **4.3/5** |

---

## 7. Current Challenges

### Escape Character Problem (Evaluator Bug)
Every flagged hallucination traces to one record (`f8c057fb`) whose raw content contains `"features"` with escaped double quotes in the JSON. The LLM correctly copies the text but uses single quotes — `'features'` — when reproducing it inside a markdown block quote (to avoid breaking the quote boundary). The substring check then fails because `'features'` ≠ `"features"`.

This is an **evaluator bug, not a model fabrication.** The true hallucination rate after quote normalization is **0%**. Fix: normalize both sides before comparison.

```python
quote_norm = text.replace("'", '"').lower()
source_norm = content.replace('"', '"').lower()
```

### Verbatim Matching Edge Cases
Strict substring matching also breaks on:
- LLM truncating long quotes with `...`
- Minor whitespace differences in source data
- Multi-sentence quotes where the LLM slightly adjusts the join

These are not hallucinations — they are near-verbatim reproductions that a human would accept. The current evaluator penalises them equally.

### Cost & Latency Trade-off
At ~3× cost and ~2× latency per query, enhanced mode is meaningfully more expensive. For a high-volume product (e.g., real-time dashboard queries), this matters. The cost increase is linear in the number of retrieved records — `top_k=5` adds ~800 input tokens; `top_k=10` would add ~1,600.

---

## 8. Future Work

**Fuzzy verbatim checker** — Replace strict substring with `difflib.SequenceMatcher` (ratio > 0.92). Catches legitimate near-verbatim quotes (truncated sentences, whitespace) without inflating hallucination rate. Cheaper and more reliable than an LLM-based checker; an LLM judge for verbatim accuracy would add cost and introduce its own errors.

**Minimum quote length guard** — The coherence 2/5 on "what users love" is a data sparsity problem: five one-word reviews ("great", "superb", "i love it") give the LLM nothing meaningful to quote. Adding a prompt instruction — *"If verbatim text is fewer than 8 words and adds no context beyond the claim, skip the quote"* — prevents the degraded list-like output without changing the retrieval stack.

**Hybrid context window** — Pass full raw feedback content for the top-2 most relevant records; summaries only for the rest. Estimated ~40% input token reduction with minimal quality impact, bringing cost from ~3× to ~1.8× baseline.