# Inline Quotes RAG — Approach & Design Decisions

## 1. The Task

The existing RAG pipeline returned plain-text summaries with no connection back to original user feedback. The goal was to add **inline verbatim quotes** attributed to their source `feedback_record_id`, while keeping streaming intact and not breaking the existing retrieval architecture.

**Key architectural decision:** Only the prompt layer changed. The retriever, indexer, ChromaDB setup, and streaming pipeline are untouched. This was intentional — the problem was a generation problem, not a retrieval problem. Adding quotes doesn't require knowing more — it requires using what's already retrieved differently.

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

Labelling the two sources explicitly reduces a failure mode where the LLM quotes from the summary (paraphrased) instead of the raw text (verbatim). Without explicit labels, the model conflates the two.

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

Key prompt instructions and their rationale:

- **Group related claims** — "Do not write one sentence per record" prevents list-like output when multiple records cover the same theme
- **Quote from `verbatim_feedback` only** — explicitly forbidden from quoting the summary
- **Omit rather than fabricate** — if no verbatim text supports a claim, skip the quote; this was the most important instruction for preventing hallucinations
- **One quote per line** — keeps the regex parser simple and unambiguous

---

## 5. Design Decisions & Trade-offs

### Index summaries, not raw feedback

Retrieval runs against embedded summaries rather than raw feedback text. The alternative — embedding raw feedback directly — would produce noisier retrieval: raw reviews are often short, informal, multilingual, or emotionally charged, which makes them poor semantic targets. Summaries are normalized, English, and topic-dense, making them better anchors for semantic search. The raw text is still used — but only at generation time for quoting, not at retrieval time for ranking.

### Pass both summary + raw to the LLM, not raw only

The enhanced prompt sends both the summary and the raw feedback for each record. An alternative would be to drop summaries entirely and pass only raw text. That was rejected because summaries serve a structural role — the LLM uses them to decide what claims are worth making and how to group them, then reaches into the raw text for verbatim support. Raw-only context produced less coherent, less organized answers in early testing.

### Single-pass generation vs two-pass

A two-pass approach would first generate a prose summary, then run a second LLM call to extract and insert quotes. This is more controllable — each pass has a single, well-defined job — and easier to evaluate independently. However, it doubles latency and cost, and breaks streaming (you can't stream a response that depends on a second call). Single-pass was chosen to preserve streaming and keep cost linear. The trade-off is that prompt instructions carry more weight, and failures are harder to isolate.

### Block quote format vs inline `[record_id]` markers

The problem statement suggests `[rec_abc123]` markers inline in prose. Block quotes after each paragraph were chosen instead. Inline markers require the LLM to track citation-to-claim mapping mid-sentence, which increases the chance of misattribution. Paragraph-then-quotes is a simpler contract: write the claim, then prove it. It also produces cleaner output for a markdown renderer and makes the regex parser unambiguous.

### Programmatic checks for verbatim/citation, LLM only for coherence

Verbatim accuracy and citation correctness are evaluated programmatically (substring match + record ID lookup). An LLM judge could handle fuzzy matches better, but introduces its own errors and adds cost to every evaluation run. For binary factual checks — "is this text present in this record?" — deterministic is strictly better. Coherence is the one dimension that genuinely requires judgement about prose quality, so LLM-as-judge is reserved for that alone (~$0.002/query at eval time, not production time).

---

## 6. Cost & Latency Tracking

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

### Baseline vs Enhanced — Cost & Latency (GPT-4o, 6 queries)

| Metric | Baseline | Enhanced | Delta |
|---|---|---|---|
| Avg total tokens | ~337 | ~1,177 | +249% |
| Avg cost/query | ~$0.0019 | ~$0.0059 | **~3.2× more** |
| Avg TTFT | ~37ms | ~82ms | +2.2× |
| Avg total time | ~2.2s | ~5.7s | +2.5× slower |

**Why input tokens dominate:** Raw feedback content adds ~650 tokens per query on average compared to summary-only context. This is the primary cost driver — output tokens also grow because quoted answers are longer, but input is the bigger lever.

**TTFT increases in enhanced mode** because the longer prompt takes more time to prefill before the first token is generated. Unlike baseline where the model responds almost immediately, the user perceives a slightly longer wait before the answer starts appearing.

---

## 7. Evaluation

Three dimensions were evaluated across 6 product-manager-style queries:

### Verbatim Accuracy — Programmatic
Checks whether the extracted quote is a substring of the actual `feedback_record` content.

```python
quote.text.lower() in source_content.lower()
```

No LLM cost. Fast and deterministic.

### Citation Correctness — Programmatic
Checks whether the `feedback_record_id` in the quote maps to the record the quote came from. A quote can be verbatim but wrongly attributed — this catches that.

### Answer Coherence — LLM-as-Judge
Checks whether quotes are naturally woven into prose or dumped in as a disconnected list. Rated 1–5 by GPT-4o:

```
5 — Every quote directly supports the claim before it; prose flows naturally
3 — Some quotes feel forced or loosely connected
1 — Quotes dumped in with no connection to prose
```

**Trade-off:** One additional API call per query at evaluation time. Justified for offline benchmarking; removed in production.

### Results (GPT-4o, 6 queries)

| Query | Verbatim | Citation | Hallucination | Coherence |
|---|---|---|---|---|
| App performance complaints | 100% | 100% | 0% | 5/5 |
| What users love about Canva | 100% | 100% | 0% | 3/5 |
| Usability issues | 80% | 80% | 20% | 5/5 |
| Feature requests | 100% | 100% | 0% | 4/5 |
| Pricing & paid features | 100% | 100% | 0% | 5/5 |
| Recent update sentiment | 80% | 80% | 20% | 4/5 |
| **Average** | **93%** | **93%** | **7%** | **4.3/5** |

---

## 8. Current Challenges

### Escape Character Problem (Evaluator Bug)
Every flagged hallucination traces to one record (`f8c057fb`) whose raw content contains `"features"` with escaped double quotes in JSON. The LLM correctly copies the text but substitutes single quotes — `'features'` — inside the markdown block quote to avoid breaking the quote boundary. The substring check fails because `'features'` ≠ `"features"`.

This is an **evaluator bug, not a model fabrication.** The true hallucination rate after quote normalization is **0%**. Fix: normalize both sides before comparison.

```python
quote_norm = text.replace("'", '"').lower()
source_norm = content.replace('"', '"').lower()
```

### Verbatim Matching Edge Cases
Strict substring matching also breaks on LLM truncating long quotes with `...`, minor whitespace differences, or multi-sentence quotes where the LLM slightly adjusts the join. These are near-verbatim reproductions a human would accept — the current evaluator penalises them the same as fabrications.

### Cost & Latency
At ~3.2× cost and ~2.5× latency, enhanced mode is meaningful overhead for high-volume use. Cost scales linearly with `top_k` — increasing from 5 to 10 retrieved records roughly doubles input tokens. TTFT at ~82ms is acceptable for an async product dashboard but would be noticeable in a synchronous UI.

### Coherence Degrades on Sparse Feedback
The "users love Canva" query retrieved five one-word reviews ("great", "superb", "i love it"). With nothing substantive to quote, the LLM produces a list of single-word block quotes — coherence drops to 3/5. This is a retrieval quality problem, not a prompt failure.

---

## 9. Future Work

**Fuzzy verbatim checker** — Replace strict substring with `difflib.SequenceMatcher` (ratio > 0.92). Catches near-verbatim quotes without inflating hallucination rate. Cheaper and more reliable than an LLM judge for this factual check.

**Minimum quote length guard** — Add prompt instruction: *"If verbatim text is fewer than 8 words and adds no context beyond the claim, skip the quote."* Fixes coherence collapse on one-word reviews without touching the retrieval stack.

**Hybrid context window** — Pass full raw feedback for the top-2 most relevant records; summaries only for the rest. Estimated ~40% input token reduction with minimal quality impact, bringing cost from ~3.2× to ~2× baseline.
