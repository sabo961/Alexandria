"""
Alexandria RAG Query Engine
============================

Unified RAG (Retrieval-Augmented Generation) system for querying the Alexandria knowledge base.

FEATURES
--------
- Semantic search via Qdrant vector database
- Similarity threshold filtering (removes irrelevant results)
- Optional LLM-based reranking (improves result quality)
- OpenRouter integration for answer generation
- CLI and Python module interfaces

BASIC USAGE (CLI)
-----------------
Simple query:
    python rag_query.py "What does Silverston say about shipments?"

With similarity threshold:
    python rag_query.py "database design" --threshold 0.6

Filter by book:
    python rag_query.py "cognitive load" --book "Thinking, Fast and Slow"

ADVANCED USAGE (CLI)
--------------------
NOTE: LLM features (--rerank, --answer) require OPENROUTER_API_KEY environment variable.
      Set it before running: export OPENROUTER_API_KEY="sk-or-v1-..."

Enable LLM reranking (better relevance):
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "shipment patterns" --rerank

Generate LLM answer (full RAG):
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "What is normalization?" --answer

Use specific model:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "cognitive load" --answer --model "gpt-4o-mini"

Combine all features:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "database design patterns" \
        --threshold 0.6 \
        --rerank \
        --answer \
        --model "gpt-4o-mini" \
        --format json

PYTHON MODULE USAGE
-------------------
Simple retrieval:
    from rag_query import perform_rag_query

    result = perform_rag_query(
        query="What is cognitive load?",
        threshold=0.5
    )

    for chunk in result.results:
        print(f"{chunk['book_title']}: {chunk['text'][:100]}")

Full RAG with reranking:
    result = perform_rag_query(
        query="Explain database normalization",
        threshold=0.6,
        enable_reranking=True,
        rerank_model="meta-llama/llama-3.2-3b-instruct:free",
        generate_llm_answer=True,
        answer_model="gpt-4o-mini",
        openrouter_api_key=api_key
    )

    print(result.answer)
    print(f"Sources: {len(result.results)} chunks")

AI AGENT USAGE
--------------
For AI agents calling this tool:

1. Simple search (no API key needed):
    result = perform_rag_query(
        query="user's question",
        collection_name="alexandria",
        limit=5,
        threshold=0.5
    )

2. With reranking (requires API key):
    result = perform_rag_query(
        query="user's question",
        threshold=0.5,
        enable_reranking=True,
        rerank_model="meta-llama/llama-3.2-3b-instruct:free",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
    )

3. Full RAG answer (requires API key):
    result = perform_rag_query(
        query="user's question",
        generate_llm_answer=True,
        answer_model="gpt-4o-mini",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    # Return answer to user
    return result.answer

CLI PARAMETERS
--------------
Required:
    query                 Natural language question

Search options:
    --collection NAME     Qdrant collection (default: alexandria)
    --limit N             Number of results (default: 5)
    --book TITLE          Filter by book title
    --threshold FLOAT     Similarity threshold 0.0-1.0 (default: 0.5)
    --fetch-multiplier N  Fetch limit×N results for filtering (default: 3, min fetch: 20)

Reranking:
    --rerank              Enable LLM reranking
    --rerank-model MODEL  Model for reranking (default: meta-llama/llama-3.2-3b-instruct:free)

Answer generation:
    --answer              Generate LLM answer
    --model MODEL         Model for answer (default: meta-llama/llama-3.2-3b-instruct:free)

API Authentication:
    OPENROUTER_API_KEY     Required for --rerank and --answer features
                           Must be set as environment variable (NOT command-line argument)

    How to set:
        export OPENROUTER_API_KEY="sk-or-v1-..."  # Linux/Mac
        set OPENROUTER_API_KEY=sk-or-v1-...       # Windows CMD
        $env:OPENROUTER_API_KEY="sk-or-v1-..."    # Windows PowerShell

    Get API key: https://openrouter.ai/keys

Output:
    --format FORMAT       Output format: markdown, text, json (default: markdown)

Qdrant:
    --host HOST           Qdrant host (from config)
    --port PORT           Qdrant port (from config)

AVAILABLE MODELS
----------------
Free models (--model or --rerank-model):
    meta-llama/llama-3.2-3b-instruct:free
    meta-llama/llama-3.2-1b-instruct:free
    qwen/qwen-2.5-7b-instruct:free
    mistralai/mistral-7b-instruct:free

Paid models (better quality):
    openai/gpt-4o-mini
    openai/gpt-4o
    anthropic/claude-3.5-sonnet
    anthropic/claude-3.5-haiku

EXAMPLES
--------
1. Basic semantic search:
    python rag_query.py "What is the shipment pattern?"

2. Search specific book:
    python rag_query.py "cognitive load" --book "Thinking, Fast and Slow" --limit 3

3. High-quality retrieval (with reranking):
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "database normalization" \
        --threshold 0.6 \
        --rerank \
        --limit 5

4. Full RAG answer:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "Explain Mishima's philosophy" \
        --answer \
        --model "gpt-4o-mini"

5. JSON output for scripting:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_query.py "data modeling patterns" \
        --answer \
        --format json > output.json

OUTPUT FORMATS
--------------
markdown  - Human-readable with sections (default)
text      - Plain text, compact
json      - Structured JSON for parsing

RETURN VALUES (Python module)
------------------------------
RAGResult object with:
    .query            - Original query string
    .results          - List of dicts with: score, book_title, author, section_name, text
    .answer           - Generated answer (if generate_llm_answer=True)
    .filtered_count   - Number of initial results before filtering
    .reranked         - Boolean, whether reranking was used

TIPS
----
- Start with threshold=0.5, adjust based on results
- Use reranking for critical queries where quality > speed
- Free models are fast and good enough for most queries
- Paid models (gpt-4o-mini) offer better answer quality
- JSON output is ideal for AI agents and scripting
- Increase --fetch-multiplier (4-5) for better quality when using reranking
- Lower --fetch-multiplier (2) for faster queries when quality is sufficient

TROUBLESHOOTING
---------------
"No results above threshold":
    → Lower --threshold value (try 0.3)
    → Check if collection has relevant content

"API key required":
    → Set OPENROUTER_API_KEY environment variable
    → Get key from https://openrouter.ai/keys

"Qdrant connection error":
    → Check if Qdrant is running (see .env for host)
    → Verify --host and --port parameters
    → Run: python scripts/configure.py --test

ARCHITECTURE
------------
This is the unified RAG query engine used by:
- CLI (this script)
- Streamlit GUI (alexandria_app.py)
- AI agents
- Python automation scripts

Single source of truth for RAG logic.
"""

import argparse
import logging
import os
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

# Import from central config
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, OPENROUTER_API_KEY

# Phoenix observability (lazy init)
try:
    from phoenix_init import get_tracer, is_phoenix_enabled
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    def get_tracer():
        return None
    def is_phoenix_enabled():
        return False
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ingest_books import generate_embeddings

# Fix Windows terminal encoding for Croatian/multilingual output
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG query"""
    query: str
    results: List[Dict[str, Any]]
    answer: Optional[str] = None
    filtered_count: int = 0
    reranked: bool = False
    initial_count: int = 0
    error: Optional[str] = None

    # Hierarchical metadata
    context_mode: str = "precise"
    parent_chunks: List[Dict[str, Any]] = field(default_factory=list)
    total_context_tokens: int = 0
    hierarchy_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Alias for results (for backward compatibility with GUI)"""
        return self.results


def get_collection_model_id(
    collection_name: str,
    host: str,
    port: int
) -> Optional[str]:
    """
    Get the embedding model ID used by a collection.

    Samples one point from the collection to detect which embedding model
    was used during ingestion. This enables automatic model selection for queries.

    Args:
        collection_name: Qdrant collection name
        host: Qdrant server host
        port: Qdrant server port

    Returns:
        Model ID string (e.g., "bge-large") if found in payload, None otherwise.
        Returns None for legacy data without embedding_model_id metadata.
    """
    try:
        client = QdrantClient(host=host, port=port)

        # Scroll to get one point from the collection
        results = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        points = results[0]  # scroll returns (points, next_offset)
        if points and len(points) > 0:
            payload = points[0].payload
            model_id = payload.get('embedding_model_id')
            if model_id:
                logger.debug(f"Detected collection model: {model_id}")
                return model_id

        logger.debug(f"No embedding_model_id found in collection '{collection_name}' (legacy data)")
        return None

    except Exception as e:
        logger.warning(f"Failed to detect collection model: {e}")
        return None


def search_qdrant(
    query: str,
    collection_name: str,
    limit: int,
    book_filter: Optional[str],
    threshold: float,
    host: str,
    port: int,
    fetch_multiplier: int = 3,
    chunk_level_filter: Optional[str] = None  # NEW: Filter by chunk_level (child/parent)
) -> tuple[List[Any], int]:
    """
    Search Qdrant and apply similarity threshold filtering.

    Automatically detects the embedding model used by the collection and uses
    the same model for query embedding generation. Falls back to default model
    for legacy collections without model metadata.

    Args:
        chunk_level_filter: If set, filter to only "child" or "parent" chunks.
                           For hierarchical retrieval, use "child" to search
                           only semantic chunks (not chapter-level parents).

    Returns:
        (filtered_results, initial_count)
    """
    import time
    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span("qdrant_search") if tracer else None
    
    if span_ctx:
        span = span_ctx.__enter__()
        span.set_attribute("query", query[:200])  # Truncate for safety
        span.set_attribute("collection", collection_name)
        span.set_attribute("limit", limit)
        span.set_attribute("threshold", threshold)
        span.set_attribute("book_filter", book_filter or "none")
        span.set_attribute("chunk_level_filter", chunk_level_filter or "none")
    
    embed_start = time.time()
    client = QdrantClient(host=host, port=port)

    # Auto-detect embedding model from collection metadata
    collection_model_id = get_collection_model_id(collection_name, host, port)
    if collection_model_id:
        logger.info(f"[MODEL] Using collection model: {collection_model_id}")
    else:
        logger.info("[MODEL] Using default model (legacy collection or no metadata)")

    # Generate query embedding using detected or default model
    logger.info(f"[SEARCH] Query: '{query}'")
    query_vector = generate_embeddings([query], model_id=collection_model_id)[0]
    embed_time = time.time() - embed_start
    
    if span_ctx:
        span.set_attribute("embedding_time_ms", int(embed_time * 1000))

    # Build filter
    conditions = []
    if book_filter:
        conditions.append(
            FieldCondition(key="book_title", match=MatchValue(value=book_filter))
        )
        logger.info(f"[BOOK] Filtering by book: {book_filter}")

    # NEW: Filter by chunk level (for hierarchical retrieval)
    if chunk_level_filter:
        conditions.append(
            FieldCondition(key="chunk_level", match=MatchValue(value=chunk_level_filter))
        )
        logger.info(f"[FILTER] Filtering by chunk_level: {chunk_level_filter}")

    query_filter = Filter(must=conditions) if conditions else None

    # Fetch more results than needed for better filtering/reranking
    # fetch_multiplier controls how many extra results to retrieve
    fetch_limit = max(20, limit * fetch_multiplier)

    # Search
    search_start = time.time()
    initial_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_limit,
        query_filter=query_filter
    ).points
    search_time = time.time() - search_start

    logger.info(f"[OK] Retrieved {len(initial_results)} initial results")

    # Apply similarity threshold filter
    filtered_results = [r for r in initial_results if r.score >= threshold]

    if len(filtered_results) < len(initial_results):
        logger.info(f"🎯 Filtered to {len(filtered_results)} results above threshold ({threshold:.2f})")

    # Record Phoenix metrics
    if span_ctx:
        span.set_attribute("search_time_ms", int(search_time * 1000))
        span.set_attribute("initial_results", len(initial_results))
        span.set_attribute("filtered_results", len(filtered_results))
        if filtered_results:
            span.set_attribute("top_score", filtered_results[0].score)
        span_ctx.__exit__(None, None, None)

    return filtered_results, len(initial_results)


def fetch_parent_chunks(
    child_results: List[Any],
    collection_name: str,
    client: QdrantClient
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch parent chunk content for hierarchical context.

    Args:
        child_results: List of child chunk search results
        collection_name: Qdrant collection name
        client: Qdrant client instance

    Returns:
        Dict mapping parent_id -> parent chunk data (text, title, full_text)
    """
    # Collect unique parent IDs from child results
    parent_ids = set()
    for result in child_results:
        parent_id = result.payload.get('parent_id')
        if parent_id:
            parent_ids.add(parent_id)

    if not parent_ids:
        logger.debug("No parent IDs found in child results")
        return {}

    logger.info(f"[BOOK] Fetching {len(parent_ids)} parent chunks for context...")

    # Fetch parent chunks by ID
    parent_chunks = {}
    try:
        # Retrieve points by their IDs
        points = client.retrieve(
            collection_name=collection_name,
            ids=list(parent_ids),
            with_payload=True
        )

        for point in points:
            parent_chunks[str(point.id)] = {
                'id': str(point.id),
                'text': point.payload.get('text', ''),
                'full_text': point.payload.get('full_text', point.payload.get('text', '')),
                'section_name': point.payload.get('section_name', 'Unknown'),
                'book_title': point.payload.get('book_title', 'Unknown'),
                'author': point.payload.get('author', 'Unknown'),
                'chunk_level': point.payload.get('chunk_level', 'parent')
            }

        logger.info(f"[OK] Retrieved {len(parent_chunks)} parent chunks")

    except Exception as e:
        logger.warning(f"Failed to fetch parent chunks: {e}")

    return parent_chunks


def rerank_with_llm(
    results: List[Any],
    query: str,
    limit: int,
    rerank_model: str,
    openrouter_api_key: str
) -> List[Any]:
    """
    Rerank results using LLM relevance scoring.

    Args:
        results: Initial search results
        query: User query
        limit: Final number of results to return
        rerank_model: OpenRouter model ID
        openrouter_api_key: API key

    Returns:
        Reranked results (top N)
    """
    import requests
    import time
    
    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span("llm_rerank") if tracer else None
    
    if span_ctx:
        span = span_ctx.__enter__()
        span.set_attribute("model", rerank_model)
        span.set_attribute("input_count", len(results))
        span.set_attribute("limit", limit)
    
    rerank_start = time.time()
    logger.info(f"🤖 Reranking top {min(10, len(results))} results with {rerank_model}...")

    rerank_scores = []
    for result in results[:10]:  # Only rerank top 10 to save API calls
        rerank_prompt = f"""Rate the relevance of this text to the question on a scale of 0-10.

Question: {query}

Text: {result.payload.get('text', '')[:500]}

Respond with only a number from 0-10."""

        try:
            rerank_response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": rerank_model,
                    "messages": [{"role": "user", "content": rerank_prompt}]
                },
                timeout=30
            )

            if rerank_response.status_code == 200:
                score_text = rerank_response.json()["choices"][0]["message"]["content"]
                score = float(score_text.strip())
                rerank_scores.append((result, score))
            else:
                logger.warning(f"Rerank API error {rerank_response.status_code}, using default score")
                rerank_scores.append((result, 5.0))
        except (ValueError, KeyError, IndexError, Exception) as e:
            logger.warning(f"Rerank parse error: {e}, using default score")
            rerank_scores.append((result, 5.0))

    # Sort by rerank score and take top results
    rerank_scores.sort(key=lambda x: x[1], reverse=True)
    reranked_results = [r[0] for r in rerank_scores[:limit]]
    
    rerank_time = time.time() - rerank_start
    
    if span_ctx:
        span.set_attribute("rerank_time_ms", int(rerank_time * 1000))
        span.set_attribute("output_count", len(reranked_results))
        span_ctx.__exit__(None, None, None)

    logger.info(f"[OK] Reranked to top {len(reranked_results)} most relevant chunks")
    return reranked_results


def generate_answer(
    query: str,
    results: List[Any],
    model: str,
    openrouter_api_key: str,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate answer using OpenRouter LLM.

    Args:
        query: User query
        results: Retrieved context chunks
        model: OpenRouter model ID
        openrouter_api_key: API key
        temperature: Controls randomness (0.0-2.0)
        system_prompt: Optional custom system prompt (overrides default)

    Returns:
        Generated answer text
    """
    import requests
    import time
    
    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span("llm_answer") if tracer else None
    
    if span_ctx:
        span = span_ctx.__enter__()
        span.set_attribute("model", model)
        span.set_attribute("temperature", temperature)
        span.set_attribute("context_chunks", len(results))
    
    answer_start = time.time()

    # Build RAG context
    context_parts = []
    for idx, result in enumerate(results, 1):
        p = result.payload
        context_parts.append(
            f"[Source {idx}] Book: {p.get('book_title', 'Unknown')} by {p.get('author', 'Unknown')}\n"
            f"Content: {p.get('text', '')}\n"
        )

    rag_context = "\n---\n".join(context_parts)

    # Build prompt
    default_system_prompt = (
        "You are Alexandria, a helpful assistant with access to a knowledge base of books. "
        "Answer the user's question based on the provided context from the knowledge base. "
        "If the context doesn't contain enough information, say so. "
        "Always cite your sources by mentioning the book title and author."
    )
    effective_system_prompt = system_prompt if system_prompt else default_system_prompt

    user_prompt = f"Context from knowledge base:\n\n{rag_context}\n\nUser Question: {query}\n\nPlease provide a comprehensive answer based on the context above."

    logger.info(f"🤖 Generating answer with {model}...")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/alexandria",
            "X-Title": "Alexandria RAG System"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        },
        timeout=60
    )

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        answer_time = time.time() - answer_start
        
        if span_ctx:
            span.set_attribute("answer_time_ms", int(answer_time * 1000))
            span.set_attribute("answer_length", len(answer))
            span_ctx.__exit__(None, None, None)
        
        logger.info("[OK] Answer generated successfully")
        return answer
    else:
        error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
        
        if span_ctx:
            span.set_attribute("error", error_msg)
            span_ctx.__exit__(None, None, None)
        
        logger.error(error_msg)
        raise Exception(error_msg)


def perform_rag_query(
    query: str,
    collection_name: str = 'alexandria',
    limit: int = 5,
    book_filter: Optional[str] = None,
    threshold: float = 0.5,
    enable_reranking: bool = False,
    rerank_model: Optional[str] = None,
    generate_llm_answer: bool = False,
    answer_model: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    fetch_multiplier: int = 3,
    temperature: float = 0.7,
    context_mode: Literal["precise", "contextual", "comprehensive"] = "precise",
    system_prompt: Optional[str] = None
) -> RAGResult:
    """
    Unified RAG query function with hierarchical context support.

    Args:
        query: Natural language question
        collection_name: Qdrant collection to search
        limit: Number of final results to return
        book_filter: Optional filter by book title
        threshold: Similarity score threshold (0.0-1.0)
        enable_reranking: Whether to rerank with LLM
        rerank_model: Model for reranking (required if enable_reranking=True)
        generate_llm_answer: Whether to generate answer with LLM
        answer_model: Model for answer generation (required if generate_llm_answer=True)
        openrouter_api_key: OpenRouter API key (required if using LLM features)
        host: Qdrant host
        port: Qdrant port
        fetch_multiplier: How many times 'limit' to fetch from Qdrant (default: 3)
                         Fetches limit * fetch_multiplier results for better filtering/reranking
                         Min fetch is always 20 to ensure quality pool
        context_mode: Hierarchical context retrieval mode:
                     - "precise": Return only matched child chunks (default, fastest)
                     - "contextual": Include parent chapter context for each match
                     - "comprehensive": Include parent + sibling chunks (most context)
        system_prompt: Optional custom system prompt for answer generation (overrides default)

    Returns:
        RAGResult object with results and optional answer
    """
    # Validate inputs
    if enable_reranking and not rerank_model:
        raise ValueError("rerank_model required when enable_reranking=True")
    if generate_llm_answer and not answer_model:
        raise ValueError("answer_model required when generate_llm_answer=True")
    if (enable_reranking or generate_llm_answer) and not openrouter_api_key:
        raise ValueError("openrouter_api_key required for LLM features")

    # Step 1: Search Qdrant with threshold filtering
    # For hierarchical collections, search only child chunks (semantic segments)
    # Parent chunks (chapters) are retrieved separately for context
    chunk_level = "child" if context_mode != "precise" else None

    try:
        filtered_results, initial_count = search_qdrant(
            query=query,
            collection_name=collection_name,
            limit=limit,
            book_filter=book_filter,
            threshold=threshold,
            host=host,
            port=port,
            fetch_multiplier=fetch_multiplier,
            chunk_level_filter=chunk_level
        )
    except Exception as e:
        logger.error(f"Qdrant search failed: {str(e)}")
        return RAGResult(
            query=query,
            results=[],
            filtered_count=0,
            initial_count=0,
            reranked=False,
            error=f"Qdrant search failed: {str(e)}"
        )

    # Step 1b: Fetch parent chunks for contextual/comprehensive modes
    parent_chunks = {}
    if context_mode in ("contextual", "comprehensive") and filtered_results:
        client = QdrantClient(host=host, port=port)
        parent_chunks = fetch_parent_chunks(filtered_results, collection_name, client)

    if len(filtered_results) == 0:
        logger.warning(f"[WARN] No results above similarity threshold {threshold:.2f}")
        return RAGResult(
            query=query,
            results=[],
            filtered_count=0,
            initial_count=initial_count,
            reranked=False
        )

    # Step 2: Optional LLM reranking
    if enable_reranking:
        try:
            final_results = rerank_with_llm(
                results=filtered_results,
                query=query,
                limit=limit,
                rerank_model=rerank_model,
                openrouter_api_key=openrouter_api_key
            )
            reranked = True
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Fall back to filtered results without reranking
            final_results = filtered_results[:limit]
            reranked = False
    else:
        final_results = filtered_results[:limit]
        reranked = False

    # Convert results to dict format with hierarchical context
    results_data = []
    total_context_tokens = 0

    for result in final_results:
        p = result.payload
        result_dict = {
            'score': result.score,
            'book_title': p.get('book_title', 'Unknown'),
            'author': p.get('author', 'Unknown'),
            'section_name': p.get('section_name', 'Unknown'),
            'text': p.get('text', ''),
            'chunk_level': p.get('chunk_level', 'unknown'),
            'parent_id': p.get('parent_id')
        }

        # Add parent context if available
        parent_id = p.get('parent_id')
        if parent_id and parent_id in parent_chunks:
            parent = parent_chunks[parent_id]
            result_dict['parent_context'] = {
                'section_name': parent.get('section_name', 'Unknown'),
                'text': parent.get('text', ''),  # Truncated for embedding
                'full_text': parent.get('full_text', '')  # Full chapter text
            }
            # Estimate tokens (rough: 1.3 tokens per word)
            total_context_tokens += len(parent.get('full_text', '').split()) * 1.3

        results_data.append(result_dict)

    # Build hierarchy stats
    hierarchy_stats = {
        'context_mode': context_mode,
        'parent_chunks_fetched': len(parent_chunks),
        'unique_chapters': len(set(r.get('parent_id') for r in results_data if r.get('parent_id')))
    }

    # Step 3: Optional answer generation
    answer = None
    if generate_llm_answer:
        try:
            answer = generate_answer(
                query=query,
                results=final_results,
                model=answer_model,
                openrouter_api_key=openrouter_api_key,
                temperature=temperature,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return RAGResult(
                query=query,
                results=results_data,
                answer=None,
                filtered_count=len(filtered_results),
                initial_count=initial_count,
                reranked=reranked,
                error=str(e),
                context_mode=context_mode,
                parent_chunks=list(parent_chunks.values()),
                total_context_tokens=int(total_context_tokens),
                hierarchy_stats=hierarchy_stats
            )

    return RAGResult(
        query=query,
        results=results_data,
        answer=answer,
        filtered_count=len(filtered_results),
        initial_count=initial_count,
        reranked=reranked,
        context_mode=context_mode,
        parent_chunks=list(parent_chunks.values()),
        total_context_tokens=int(total_context_tokens),
        hierarchy_stats=hierarchy_stats
    )


def print_results(result: RAGResult, format: str = 'markdown'):
    """Print RAG results in specified format"""
    if format == 'json':
        import json
        output = {
            'query': result.query,
            'filtered_count': result.filtered_count,
            'reranked': result.reranked,
            'results': result.results,
            'answer': result.answer
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    elif format == 'markdown':
        print("\n" + "=" * 80)
        print("# Alexandria RAG Results")
        print("=" * 80)
        print(f"\n**Query:** {result.query}")
        print(f"**Retrieved:** {len(result.results)} chunks (filtered from {result.filtered_count})")
        print(f"**Reranked:** {'Yes' if result.reranked else 'No'}")
        print(f"**Context Mode:** {result.context_mode}")
        if result.hierarchy_stats:
            stats = result.hierarchy_stats
            print(f"**Chapters:** {stats.get('unique_chapters', 0)} unique chapters, {stats.get('parent_chunks_fetched', 0)} parent chunks fetched")

        if result.answer:
            print("\n## Answer\n")
            print(result.answer)

        print("\n## Sources\n")
        for idx, r in enumerate(result.results, 1):
            print(f"### Source {idx} (Relevance: {r['score']:.4f})")
            print(f"- **Book:** {r['book_title']}")
            print(f"- **Author:** {r['author']}")
            print(f"- **Section:** {r['section_name']}")

            # Show chapter context if available (contextual/comprehensive mode)
            if r.get('parent_context'):
                parent = r['parent_context']
                print(f"- **Chapter:** {parent.get('section_name', 'Unknown')}")

            print(f"\n> {r['text'][:500]}{'...' if len(r['text']) > 500 else ''}\n")
            print("---\n")

    else:  # text
        print(f"\nQuery: {result.query}")
        print(f"Results: {len(result.results)} chunks\n")

        if result.answer:
            print("ANSWER:")
            print(result.answer)
            print("\n" + "-" * 80 + "\n")

        for idx, r in enumerate(result.results, 1):
            print(f"[{idx}] {r['book_title']} by {r['author']} (Score: {r['score']:.4f})")
            print(r['text'][:300] + "...\n")


def main():
    parser = argparse.ArgumentParser(description='Alexandria RAG Query Engine')

    # Required
    parser.add_argument('query', type=str, help='Natural language question')

    # Search options
    parser.add_argument('--collection', type=str, default='alexandria', help='Qdrant collection')
    parser.add_argument('--limit', type=int, default=5, help='Number of results')
    parser.add_argument('--book', type=str, help='Filter by book title')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold (0.0-1.0)')
    parser.add_argument('--fetch-multiplier', type=int, default=3,
                       help='Fetch limit*N results from Qdrant for better filtering (default: 3)')

    # Reranking
    parser.add_argument('--rerank', action='store_true', help='Enable LLM reranking')
    parser.add_argument('--rerank-model', type=str, default='meta-llama/llama-3.2-3b-instruct:free',
                       help='Model for reranking')

    # Answer generation
    parser.add_argument('--answer', action='store_true', help='Generate LLM answer')
    parser.add_argument('--model', type=str, default='meta-llama/llama-3.2-3b-instruct:free',
                       help='Model for answer generation')

    # Output
    parser.add_argument('--format', type=str, choices=['markdown', 'text', 'json'],
                       default='markdown', help='Output format')

    # Hierarchical context
    parser.add_argument('--context-mode', type=str, choices=['precise', 'contextual', 'comprehensive'],
                       default='precise', help='Context retrieval mode: precise (child chunks only), '
                       'contextual (+ chapter context), comprehensive (+ siblings)')

    # Qdrant
    parser.add_argument('--host', type=str, default=QDRANT_HOST, help=f'Qdrant host (default: {QDRANT_HOST})')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')

    args = parser.parse_args()

    # Get API key from environment variable only
    api_key = os.environ.get('OPENROUTER_API_KEY')

    try:
        result = perform_rag_query(
            query=args.query,
            collection_name=args.collection,
            limit=args.limit,
            book_filter=args.book,
            threshold=args.threshold,
            enable_reranking=args.rerank,
            rerank_model=args.rerank_model if args.rerank else None,
            generate_llm_answer=args.answer,
            answer_model=args.model if args.answer else None,
            openrouter_api_key=api_key,
            host=args.host,
            port=args.port,
            fetch_multiplier=args.fetch_multiplier,
            context_mode=args.context_mode
        )

        print_results(result, format=args.format)

    except Exception as e:
        logger.error(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
