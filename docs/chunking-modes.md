# Alexandria Chunking Modes

## Overview

Alexandria supports two chunking strategies. Both produce compatible embeddings that can coexist in the same Qdrant collection.

## Fixed Chunking (Default)

**Speed:** ~10-20 books/minute  
**Use for:** Bulk ingestion, most content

```
chunk_size=500 words, overlap=50 words
```

### When to use:
- Bulk ingestion (1000+ books)
- Compilations, anthologies, zbornici
- Web scrape material (URLs in metadata)
- Manuals, how-to guides, technical docs
- Short documents (<30 pages)
- Content with "selections" or "excerpts" in title
- Any content where speed > precision

## Semantic Chunking

**Speed:** ~1-3 books/minute  
**Use for:** High-value texts where retrieval quality matters

```
threshold=0.55, min_chunk=200, max_chunk=1200 words
```

### When to use:
- **Philosophy:** Platon, Aristotel, Heidegger, Nietzsche, Schopenhauer, Kant, Wittgenstein
- **Jung + depth psychology:** Jung, Hillman, von Franz, Neumann
- **Esoterica/magic:** Crowley, Regardie, Fortune, Lévi, Agrippa
- **Buddhism:** Ajahn Chah, Thich Nhat Hanh, Suzuki, Trungpa
- **Entheogens:** McKenna, Hofmann, Wasson, Shulgin, Muraresku, Grof
- **Literature:** Borges, Dostojevski, Hesse, Blake
- **Tolkien** (always)

## Implementation

### In batch_ingest.py:
```python
result = ingest_book(
    filepath=str(book_path),
    use_semantic=False  # True for semantic, False for fixed
)
```

### In ingest_books.py:
```python
def ingest_book(..., use_semantic: bool = False):
    if use_semantic:
        chunker = UniversalChunker(embedder, threshold=0.55, ...)
    else:
        chunker = FixedChunker(chunk_size=500, overlap=50, ...)
```

## Tracking

The `ingest_log` SQLite table includes `chunking_mode` column:
- `'fixed'` — fixed-size chunking
- `'semantic'` — semantic similarity chunking

Query examples:
```sql
-- Count by mode
SELECT chunking_mode, COUNT(*) FROM ingest_log GROUP BY chunking_mode;

-- List semantic-chunked books
SELECT book_title, author FROM ingest_log WHERE chunking_mode = 'semantic';

-- Books to re-ingest with semantic
SELECT book_title FROM ingest_log 
WHERE chunking_mode = 'fixed' 
AND (author LIKE '%Jung%' OR author LIKE '%Heidegger%');
```

## Mixing Modes

Both modes produce 1024-dimensional BGE-M3 embeddings. They can coexist in the same Qdrant collection without issues.

To upgrade a book from fixed to semantic:
1. Delete existing chunks: `force_reingest=True`
2. Re-ingest with `use_semantic=True`

## Performance Comparison

| Mode | Speed | Retrieval Quality | GPU Load |
|------|-------|-------------------|----------|
| Fixed | 10-20 books/min | Good (90%) | Low |
| Semantic | 1-3 books/min | Better (95-98%) | High |

For 9000 books:
- Fixed: ~8-15 hours
- Semantic: ~50-100 hours

## Error Handling

### Timeout Protection

Books that take longer than **5 minutes** to process are automatically skipped.

**Timeout log location:**
```
X:\calibre\alexandria\.qdrant\timeout_books.txt
```

**Format:**
```
2026-03-19 11:52:00 | Inner Paths to Outer Space.epub | Timeout after 300s
```

**Common causes of timeout:**
- Corrupt EPUB files
- Extremely large books (>5000 pages)
- Malformed HTML/XML in EPUB
- Encoding issues

**To retry timeout books:**
1. Check the file manually (open in Calibre, check encoding)
2. Fix or remove corrupt files
3. Run `reingest_mismatched.py` with specific book paths

### Failed Books Log

All failures (timeout + other errors) are also logged to the SQLite database:
```sql
SELECT book_title, author FROM ingest_log WHERE success = 0;
```
