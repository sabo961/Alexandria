---
title: 'Phoenix Evaluation Framework Integration'
slug: 'phoenix-eval'
created: '2026-03-20'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['Python 3.11', 'arize-phoenix', 'opentelemetry', 'Qdrant', 'sentence-transformers']
files_to_modify: ['scripts/rag_query.py', 'requirements.txt', 'scripts/eval_golden_set.py', 'config/golden_set.json']
code_patterns: ['OpenTelemetry tracing', 'Phoenix instrumentation', 'dataclass results']
test_patterns: ['Golden set queries', 'Precision@K', 'Recall@K', 'MRR']
---

# Phoenix Evaluation Framework Integration

## Overview

### Problem Statement
Alexandria RAG sustav nema način za mjerenje kvalitete retrieval rezultata. Ne znamo:
- Koliko su rezultati relevantni (precision)
- Koliko relevantnih dokumenata propuštamo (recall)
- Kako semantic vs fixed chunking utječe na kvalitetu
- Gdje su bottlenecks u query pipeline-u

### Solution
Integrirati Arize Phoenix kao lokalni observability i evaluacijski framework:
- Real-time dashboard za praćenje svakog querya
- Automatsko računanje retrieval metrika
- A/B comparison između chunking strategija
- Embedding vizualizacija i latency breakdown

### Scope

**In Scope:**
- Phoenix instalacija na Minjak (pip)
- Instrumentacija `rag_query.py` s Phoenix/OpenTelemetry tracingom
- Golden set od 50 test pitanja s poznatim odgovorima
- Dashboard za Precision@K, Recall@K, MRR metrike
- Chunking A/B comparison (semantic vs fixed)

**Out of Scope:**
- Cloud hosting (ostaje lokalno na Minjak)
- MTEB/BEIR full benchmark (koristimo objavljene rezultate za BGE-M3)
- Automatski retraining embeddings
- LLM evaluation (faithfulness, hallucination) — samo retrieval

---

## Context for Development

### Codebase Patterns

| Pattern | Usage |
|---------|-------|
| Dataclass results | `RAGResult` dataclass in rag_query.py |
| Config from module | `from config import QDRANT_HOST, QDRANT_PORT` |
| Logging | `logging.getLogger(__name__)` with INFO level |
| CLI + module | Functions usable both from CLI and as imports |

### Files to Reference

| File | Purpose |
|------|---------|
| `scripts/rag_query.py` | Main RAG entry point — add tracing here |
| `scripts/config.py` | Configuration loader — add Phoenix port |
| `requirements.txt` | Dependencies — add phoenix + otel |
| `scripts/ingest_books.py` | Reference for `generate_embeddings()` |

### Technical Decisions

1. **Phoenix over Ragas** — Phoenix provides dashboard + tracing, Ragas is metrics-only
2. **OpenTelemetry native** — Phoenix uses OTEL, future-proof
3. **Port 6006** — Standard Phoenix port, same as TensorBoard
4. **No LLM evals yet** — Focus on retrieval quality first

---

## Implementation Plan

### Task 1: Add Dependencies
- [ ] File: `requirements.txt`
- [ ] Action: Add `arize-phoenix>=4.0.0`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation`
- [ ] Notes: Pin phoenix to 4.x for stability

### Task 2: Create Phoenix Initialization Module
- [ ] File: `scripts/phoenix_init.py` (NEW)
- [ ] Action: Create module that:
  - Launches Phoenix app on port 6006
  - Configures OpenTelemetry tracer
  - Exports `get_tracer()` function
- [ ] Notes: Should be importable without side effects (lazy init)

### Task 3: Instrument rag_query.py
- [ ] File: `scripts/rag_query.py`
- [ ] Action: 
  - Import tracer from phoenix_init
  - Wrap `search_qdrant()` in span (track: query, embedding_time, search_time, results_count)
  - Wrap `rerank_with_llm()` in span (track: model, latency)
  - Wrap `generate_answer()` in span (track: model, tokens, latency)
  - Add attributes to spans: book_filter, threshold, context_mode
- [ ] Notes: Use `@tracer.start_as_current_span` decorator pattern

### Task 4: Create Golden Set
- [ ] File: `config/golden_set.json` (NEW)
- [ ] Action: Create JSON with 50 test queries:
  ```json
  {
    "queries": [
      {
        "question": "What is Thelema's core teaching?",
        "expected_books": ["Book of the Law", "Liber II"],
        "expected_authors": ["Crowley"],
        "min_relevant": 3
      }
    ]
  }
  ```
- [ ] Notes: Cover diverse topics — philosophy, psychology, occult, literature

### Task 5: Create Evaluation Script
- [ ] File: `scripts/eval_golden_set.py` (NEW)
- [ ] Action: Create script that:
  - Loads golden_set.json
  - Runs each query through perform_rag_query()
  - Computes Precision@K (how many results are from expected books)
  - Computes Recall@K (how many expected books appear in results)
  - Computes MRR (rank of first relevant result)
  - Outputs summary + per-query breakdown
  - Optionally filters by chunking_mode for A/B
- [ ] Notes: Use argparse for --limit, --threshold, --chunking-mode

### Task 6: Add Config Entry
- [ ] File: `scripts/config.py`
- [ ] Action: Add `PHOENIX_PORT = 6006` and `PHOENIX_ENABLED = True`
- [ ] Notes: Allow disabling Phoenix for production/CI

### Task 7: Update Documentation
- [ ] File: `docs/evaluation.md` (NEW)
- [ ] Action: Document:
  - How to start Phoenix dashboard
  - How to run golden set evaluation
  - How to interpret metrics
  - A/B comparison workflow

---

## Acceptance Criteria

- [ ] AC1: Given Phoenix is installed, when I run `python -c "import phoenix; phoenix.launch_app()"`, then dashboard opens at http://localhost:6006
- [ ] AC2: Given instrumentation is complete, when I run `python rag_query.py "test query"`, then a trace appears in Phoenix dashboard with timing breakdown
- [ ] AC3: Given golden_set.json has 50 queries, when I run `python eval_golden_set.py`, then I get Precision@5, Recall@5, and MRR numbers
- [ ] AC4: Given chunks have chunking_mode metadata, when I run `python eval_golden_set.py --chunking-mode semantic`, then only semantic chunks are evaluated
- [ ] AC5: Given Phoenix dashboard is running, when accessed from `http://192.168.0.229:6006`, then it's accessible from LAN

---

## Dependencies

**External:**
- `arize-phoenix` >= 4.0.0
- `opentelemetry-api`, `opentelemetry-sdk`

**Internal:**
- Qdrant running on BUCO (192.168.0.151:6333)
- Alexandria collection with chunking_mode metadata

---

## Testing Strategy

**Unit Tests:**
- Test golden_set.json schema validation
- Test metric calculation functions (precision, recall, MRR)

**Integration Tests:**
- Run eval script against live Qdrant
- Verify traces appear in Phoenix

**Manual Testing:**
- Open Phoenix dashboard, run queries, verify traces
- Compare A/B metrics between chunking modes

---

## Notes

### High-Risk Items
- Phoenix port 6006 might conflict with TensorBoard if running
- OpenTelemetry can add latency — measure baseline first

### Known Limitations
- Precision/Recall require manual golden set curation
- No ground-truth for "correct" chunks, only book-level relevance

### Future Considerations
- Add LLM-based relevance scoring (use Ragas evaluators)
- Embedding drift detection when adding new books
- Automated golden set generation from book metadata
