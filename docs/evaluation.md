# Alexandria RAG Evaluation

This document describes how to evaluate and measure the quality of Alexandria's RAG (Retrieval-Augmented Generation) system.

## Overview

Alexandria uses [Arize Phoenix](https://phoenix.arize.com/) for:
- **Observability**: Real-time tracing of every query
- **Evaluation**: Precision, Recall, MRR metrics
- **A/B Testing**: Compare semantic vs fixed chunking

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `arize-phoenix` - Observability dashboard
- `opentelemetry-*` - Tracing SDK

### 2. Start Phoenix Dashboard

```bash
# Option A: Launch from Python
python -c "import phoenix as px; px.launch_app(port=6006)"

# Option B: Launch automatically (first query triggers it)
python scripts/rag_query.py "test query"
```

Dashboard URL: `http://localhost:6006`

For LAN access (from other machines): `http://192.168.0.229:6006`

### 3. Run Evaluation

```bash
# Basic evaluation
cd scripts
python eval_golden_set.py

# With custom parameters
python eval_golden_set.py --limit 10 --threshold 0.4

# A/B test: semantic vs fixed chunking
python eval_golden_set.py --chunking-mode semantic
python eval_golden_set.py --chunking-mode fixed

# Filter by category
python eval_golden_set.py --category psychology

# JSON output for automation
python eval_golden_set.py --format json > results.json
```

## Metrics Explained

### Precision@K
**What fraction of retrieved results are relevant?**

```
Precision@5 = relevant_in_top_5 / 5
```

Higher is better. 1.0 means all retrieved results are relevant.

### Recall@K
**What fraction of relevant items were retrieved?**

```
Recall@5 = relevant_found / min_relevant_expected
```

Higher is better. 1.0 means we found all expected relevant sources.

### MRR (Mean Reciprocal Rank)
**Where does the first relevant result appear?**

```
MRR = 1 / rank_of_first_relevant
```

- First result relevant: MRR = 1.0
- Second result relevant: MRR = 0.5
- Third result relevant: MRR = 0.33

Higher is better. Shows how quickly users find relevant content.

## Golden Set

The golden set (`config/golden_set.json`) contains 50 test queries across categories:
- **philosophy** - Plato, Aristotle, Heidegger, Nietzsche, etc.
- **psychology** - Jung, Hillman, Grof, von Franz
- **literature** - Tolkien, Dostoevsky, Borges, Hesse
- **occult** - Crowley, Regardie, Fortune, Agrippa
- **psychedelics** - McKenna, Hofmann, Castaneda
- **buddhism** - Suzuki, Trungpa, Thich Nhat Hanh
- **technical** - Architecture, domain-driven design

Each query specifies:
- `expected_books` - Book titles that should appear
- `expected_authors` - Author names that should appear
- `min_relevant` - Minimum relevant results to pass

### Adding Queries

Edit `config/golden_set.json`:

```json
{
  "id": "my-query-01",
  "question": "What is the meaning of X?",
  "expected_books": ["Book Title"],
  "expected_authors": ["Author Name"],
  "min_relevant": 2,
  "category": "philosophy"
}
```

## A/B Testing: Semantic vs Fixed Chunking

Alexandria uses two chunking strategies:
- **Semantic**: Splits at sentence/paragraph boundaries (for whitelisted authors)
- **Fixed**: Splits at character count boundaries (default)

To compare:

```bash
# Run both
python eval_golden_set.py --chunking-mode semantic --format json > semantic.json
python eval_golden_set.py --chunking-mode fixed --format json > fixed.json

# Compare metrics
# (use your favorite diff tool or jq)
jq '.metrics' semantic.json fixed.json
```

## Phoenix Dashboard

### Traces View
Shows every RAG query with timing breakdown:
- `qdrant_search`: Embedding + vector search time
- `llm_rerank`: Optional LLM reranking time
- `llm_answer`: Answer generation time

### Attributes
Each trace includes:
- `query`: The user's question
- `collection`: Qdrant collection name
- `threshold`: Similarity threshold used
- `initial_results`: Total results from Qdrant
- `filtered_results`: Results above threshold
- `top_score`: Highest similarity score

### Spans
Nested spans show the pipeline:
```
rag_query (root)
├── qdrant_search
│   ├── embedding_time_ms
│   └── search_time_ms
├── llm_rerank (optional)
└── llm_answer (optional)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOENIX_ENABLED` | `true` | Enable/disable Phoenix tracing |
| `PHOENIX_PORT` | `6006` | Dashboard port |

### Disable for CI/Production

```bash
export PHOENIX_ENABLED=false
python scripts/rag_query.py "query"  # No tracing overhead
```

## Interpreting Results

### Good Results (target)
- Precision@5 ≥ 0.6 (60% of results relevant)
- Recall@5 ≥ 0.8 (80% of expected sources found)
- MRR ≥ 0.7 (first relevant usually in top 2)

### Common Issues

**Low Precision, High Recall**
- Retrieving too many irrelevant results
- Try: Increase threshold (0.5 → 0.6)

**High Precision, Low Recall**
- Missing relevant results
- Try: Decrease threshold, increase limit

**Low MRR**
- Relevant results buried in list
- Try: Enable reranking (`--rerank`)

## Troubleshooting

### Phoenix not starting
```bash
# Check if port is in use
lsof -i :6006

# Try different port
export PHOENIX_PORT=6007
```

### No traces appearing
1. Check `PHOENIX_ENABLED` is not `false`
2. Verify Phoenix is running: `http://localhost:6006`
3. Run a query and wait ~5 seconds for trace

### Import errors
```bash
pip install arize-phoenix opentelemetry-api opentelemetry-sdk
```
