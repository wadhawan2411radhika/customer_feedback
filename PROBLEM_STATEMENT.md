# AI Engineer Take-Home Assignment: Inline Quotes

## Background

This repository contains a RAG (Retrieval-Augmented Generation) system built over user feedback data. The pipeline works as follows:

1. **Indexing** — Feedback summaries are embedded using OpenAI's `text-embedding-3-small` model and stored in a ChromaDB vector database
2. **Retrieval** — User queries are embedded and matched against stored summaries using semantic search
3. **Generation** — The top-k relevant summaries are passed to an LLM, which streams a plain-text answer

Currently, the system produces plain summary answers with no connection back to the original feedback. Your task is to enhance the output with **inline quotes** — verbatim excerpts from the original feedback records, cited by their `feedback_record_id`.

## Reference Example

### Current Output

```
Users are facing DNS setup issues and 403 errors when publishing websites.
Some users report that their sites don't appear online despite being marked
as published.
```

### Expected Output

```
Users are facing DNS setup issues and 403 errors when
publishing websites. Some users report that their sites
don't appear online despite being marked as published.

> "I keep getting a 403 forbidden error every time I try to access my
> published site." — rec_def456
```

### Key Characteristics

- **Inline citations**: Summary text contains citation markers (e.g., `[rec_abc123]`) referencing the `feedback_record_id` of the source
- **Verbatim quotes**: Actual text from the original feedback record content appears as block quotes
- **Attribution**: Each quote is attributed to its source `feedback_record_id`
- **Multiple sources**: A single claim can be supported by multiple feedback records

## Requirements

1. **Inline quotes** — The LLM must produce quotes woven into the answer, not appended as a separate section at the end
2. **Citations** — Every quote must reference the `feedback_record_id` of its source feedback record
3. **Streaming** — The answer must still be streamed (chunked async output)
4. **Data awareness** — The system must pass sufficient context to the LLM (feedback record content, not just summaries) so it can extract verbatim quotes
5. **Model flexibility** — You may use any LLM provider (OpenAI, Anthropic, Google, etc.)

## Evaluation

After implementing the quotes feature, you must evaluate your solution across three dimensions:

1. **Quality** — Are the quotes accurate and actually verbatim? Are citations pointing to the correct `feedback_record_id`? Is the answer coherent with inline quotes woven in?
2. **Cost** — What is the cost per query? How does it compare to the current plain-answer approach?
3. **Latency** — What is the time-to-first-token and total response time? How does streaming performance change with the additional context?

The evaluation approach is open-ended — design your own methodology.

## Setup & Data

### Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your API key(s)
```

### Data

Data is provided separately as a zip file. Extract it into a `data/` directory at the repo root.

There are ~100 feedback records in total. Each entry is a directory (named by a UUID) containing two files:

#### `feedback_summary.json`

An LLM-generated summary of the original feedback. Key fields:

| Field | Path | Description |
|-------|------|-------------|
| `id` | `$.id` | Unique ID of this summary |
| `content` | `$.attributes.content.string.values[0]` | The summary text (this is what gets indexed and embedded) |
| `feedback_record_id` | `$.attributes.feedback_record_id.string.values[0]` | ID of the source feedback record — **use this for citations** |

#### `feedback_record.json`

The original user feedback. Key fields:

| Field | Path | Description |
|-------|------|-------------|
| `id` | `$.id` | Unique ID of this record (matches `feedback_record_id` in the summary) |
| `content` | `$.attributes.content.string.values[0]` | The verbatim user feedback text — **extract quotes from this** |
| `source` | `$.attributes.source.string.values[0]` | Where the feedback came from |
| `language` | `$.attributes.language.string.values[0]` | Language of the feedback |

#### Relationship

```
feedback_summary.feedback_record_id  ──references──>  feedback_record.id
```

The summary's `content` is a concise restatement; the record's `content` is the raw user text you should quote from.

### Running

```bash
# Step 0: Unzip the data
unzip data.zip

# Step 1: Build the vector index
uv run python main.py index --data-dir data --index-path .chroma_db

# Step 2: Query the system
uv run python main.py query "Summarize the top complaints" --model gpt-4o-mini --top-k 5
```

### Example Queries

Here are a few queries to test with (the kind a product manager would ask):

```
"What are the most common complaints about app performance?"
"Summarize what users love most about Canva"
"What usability issues are users facing?"
"What feature requests or suggestions are users making?"
"How do users feel about Canva's pricing and paid features?"
"What are users saying about the recent update?"
```

## Deliverables

1. **Modified source code** with the inline quotes feature implemented
2. **Evaluation results** — document, notebook, or script (format is your choice)
3. **Brief write-up** of your approach and design decisions

## What We're Looking For

- Clean, well-structured async Python code
- Thoughtful prompt engineering for quote extraction
- Streaming maintained throughout the pipeline
- Evaluation methodology
- Clear communication of trade-offs and design decisions
