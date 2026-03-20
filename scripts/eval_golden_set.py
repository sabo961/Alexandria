"""
Alexandria Golden Set Evaluation
=================================

Run evaluation queries against the golden set and compute retrieval metrics.

USAGE
-----
    # Basic evaluation
    python eval_golden_set.py
    
    # With custom parameters
    python eval_golden_set.py --limit 10 --threshold 0.4
    
    # Filter by chunking mode (A/B comparison)
    python eval_golden_set.py --chunking-mode semantic
    python eval_golden_set.py --chunking-mode fixed
    
    # Filter by category
    python eval_golden_set.py --category psychology
    
    # JSON output for automation
    python eval_golden_set.py --format json > results.json

METRICS
-------
    Precision@K: What fraction of retrieved results are relevant?
    Recall@K: What fraction of relevant items were retrieved?
    MRR: Mean Reciprocal Rank - where does the first relevant result appear?
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_query import perform_rag_query
from config import QDRANT_HOST, QDRANT_PORT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query_id: str
    question: str
    category: str
    
    # Retrieval results
    retrieved_books: List[str]
    retrieved_authors: List[str]
    result_scores: List[float]
    
    # Expected
    expected_books: List[str]
    expected_authors: List[str]
    min_relevant: int
    
    # Metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    relevant_count: int = 0
    is_pass: bool = False


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    total_queries: int
    passed_queries: int
    
    # Aggregate metrics
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    
    # Per-category breakdown
    category_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Configuration
    limit: int = 5
    threshold: float = 0.5
    chunking_mode: Optional[str] = None
    
    # Detailed results
    query_results: List[QueryResult] = field(default_factory=list)


def load_golden_set(path: str = None) -> Dict[str, Any]:
    """Load golden set from JSON file."""
    if path is None:
        path = Path(__file__).parent.parent / "config" / "golden_set.json"
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_relevant(result: Dict, expected_books: List[str], expected_authors: List[str], 
                title_contains: List[str] = None) -> bool:
    """
    Check if a result is relevant to the expected sources.
    
    A result is relevant if:
    - Book title matches (partial, case-insensitive) any expected book
    - Author matches (partial, case-insensitive) any expected author
    - Title contains any of the title_contains patterns
    """
    book_title = result.get('book_title', '').lower()
    author = result.get('author', '').lower()
    
    # Check book match
    for expected in expected_books:
        if expected.lower() in book_title:
            return True
    
    # Check author match
    for expected in expected_authors:
        if expected.lower() in author:
            return True
    
    # Check title contains patterns
    if title_contains:
        for pattern in title_contains:
            if pattern.lower() in book_title:
                return True
    
    return False


def compute_precision_at_k(results: List[Dict], expected_books: List[str], 
                           expected_authors: List[str], k: int,
                           title_contains: List[str] = None) -> float:
    """Compute Precision@K - fraction of retrieved that are relevant."""
    if not results:
        return 0.0
    
    results_k = results[:k]
    relevant = sum(1 for r in results_k if is_relevant(r, expected_books, expected_authors, title_contains))
    return relevant / len(results_k)


def compute_recall_at_k(results: List[Dict], expected_books: List[str],
                        expected_authors: List[str], min_relevant: int,
                        title_contains: List[str] = None) -> float:
    """
    Compute Recall@K - fraction of relevant items that were retrieved.
    
    Since we don't have ground truth for ALL relevant chunks, we use min_relevant
    as the expected number of unique relevant results.
    """
    if min_relevant == 0:
        return 1.0
    
    relevant = sum(1 for r in results if is_relevant(r, expected_books, expected_authors, title_contains))
    return min(1.0, relevant / min_relevant)


def compute_mrr(results: List[Dict], expected_books: List[str],
                expected_authors: List[str], title_contains: List[str] = None) -> float:
    """Compute Mean Reciprocal Rank - 1/rank of first relevant result."""
    for i, result in enumerate(results):
        if is_relevant(result, expected_books, expected_authors, title_contains):
            return 1.0 / (i + 1)
    return 0.0


def evaluate_query(query_spec: Dict, limit: int, threshold: float,
                   host: str, port: int, chunking_mode: Optional[str] = None) -> QueryResult:
    """Evaluate a single query from the golden set."""
    question = query_spec['question']
    query_id = query_spec['id']
    category = query_spec.get('category', 'unknown')
    
    expected_books = query_spec.get('expected_books', [])
    expected_authors = query_spec.get('expected_authors', [])
    title_contains = query_spec.get('title_contains', [])
    min_relevant = query_spec.get('min_relevant', 1)
    
    logger.info(f"[EVAL] {query_id}: {question[:50]}...")
    
    # Run RAG query
    try:
        result = perform_rag_query(
            query=question,
            collection_name='alexandria',
            limit=limit,
            threshold=threshold,
            host=host,
            port=port
        )
        
        results = result.results
        
        # TODO: Filter by chunking_mode if specified
        # This requires chunking_mode to be in result payload
        if chunking_mode:
            results = [r for r in results if r.get('chunking_mode') == chunking_mode]
        
    except Exception as e:
        logger.error(f"[EVAL] {query_id} failed: {e}")
        return QueryResult(
            query_id=query_id,
            question=question,
            category=category,
            retrieved_books=[],
            retrieved_authors=[],
            result_scores=[],
            expected_books=expected_books,
            expected_authors=expected_authors,
            min_relevant=min_relevant,
            is_pass=False
        )
    
    # Extract retrieved info
    retrieved_books = [r.get('book_title', 'Unknown') for r in results]
    retrieved_authors = [r.get('author', 'Unknown') for r in results]
    result_scores = [r.get('score', 0.0) for r in results]
    
    # Compute metrics
    precision = compute_precision_at_k(results, expected_books, expected_authors, limit, title_contains)
    recall = compute_recall_at_k(results, expected_books, expected_authors, min_relevant, title_contains)
    mrr = compute_mrr(results, expected_books, expected_authors, title_contains)
    
    relevant_count = sum(1 for r in results if is_relevant(r, expected_books, expected_authors, title_contains))
    is_pass = relevant_count >= min_relevant
    
    return QueryResult(
        query_id=query_id,
        question=question,
        category=category,
        retrieved_books=retrieved_books,
        retrieved_authors=retrieved_authors,
        result_scores=result_scores,
        expected_books=expected_books,
        expected_authors=expected_authors,
        min_relevant=min_relevant,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=mrr,
        relevant_count=relevant_count,
        is_pass=is_pass
    )


def run_evaluation(
    golden_set_path: str = None,
    limit: int = 5,
    threshold: float = 0.5,
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    chunking_mode: Optional[str] = None,
    category_filter: Optional[str] = None
) -> EvalSummary:
    """Run full evaluation on golden set."""
    
    # Load golden set
    golden_set = load_golden_set(golden_set_path)
    queries = golden_set['queries']
    
    # Filter by category if specified
    if category_filter:
        queries = [q for q in queries if q.get('category') == category_filter]
        logger.info(f"[EVAL] Filtered to {len(queries)} queries in category '{category_filter}'")
    
    # Evaluate each query
    results = []
    for query_spec in queries:
        result = evaluate_query(
            query_spec=query_spec,
            limit=limit,
            threshold=threshold,
            host=host,
            port=port,
            chunking_mode=chunking_mode
        )
        results.append(result)
    
    # Aggregate metrics
    if results:
        mean_precision = sum(r.precision_at_k for r in results) / len(results)
        mean_recall = sum(r.recall_at_k for r in results) / len(results)
        mean_mrr = sum(r.reciprocal_rank for r in results) / len(results)
        passed = sum(1 for r in results if r.is_pass)
    else:
        mean_precision = mean_recall = mean_mrr = 0.0
        passed = 0
    
    # Per-category breakdown
    category_results = {}
    categories = set(r.category for r in results)
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            category_results[cat] = {
                'count': len(cat_results),
                'precision': sum(r.precision_at_k for r in cat_results) / len(cat_results),
                'recall': sum(r.recall_at_k for r in cat_results) / len(cat_results),
                'mrr': sum(r.reciprocal_rank for r in cat_results) / len(cat_results),
                'pass_rate': sum(1 for r in cat_results if r.is_pass) / len(cat_results)
            }
    
    return EvalSummary(
        total_queries=len(results),
        passed_queries=passed,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        mean_mrr=mean_mrr,
        category_results=category_results,
        limit=limit,
        threshold=threshold,
        chunking_mode=chunking_mode,
        query_results=results
    )


def print_summary(summary: EvalSummary, format: str = 'text'):
    """Print evaluation summary."""
    
    if format == 'json':
        output = {
            'total_queries': summary.total_queries,
            'passed_queries': summary.passed_queries,
            'pass_rate': summary.passed_queries / summary.total_queries if summary.total_queries else 0,
            'metrics': {
                'precision_at_k': summary.mean_precision,
                'recall_at_k': summary.mean_recall,
                'mrr': summary.mean_mrr
            },
            'config': {
                'limit': summary.limit,
                'threshold': summary.threshold,
                'chunking_mode': summary.chunking_mode
            },
            'category_breakdown': summary.category_results,
            'query_results': [
                {
                    'id': r.query_id,
                    'question': r.question,
                    'category': r.category,
                    'precision': r.precision_at_k,
                    'recall': r.recall_at_k,
                    'mrr': r.reciprocal_rank,
                    'relevant_found': r.relevant_count,
                    'min_required': r.min_relevant,
                    'pass': r.is_pass
                }
                for r in summary.query_results
            ]
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return
    
    # Text format
    print("\n" + "=" * 70)
    print("ALEXANDRIA RAG EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 CONFIGURATION")
    print(f"   Limit: {summary.limit}")
    print(f"   Threshold: {summary.threshold}")
    if summary.chunking_mode:
        print(f"   Chunking Mode: {summary.chunking_mode}")
    
    print(f"\n📈 AGGREGATE METRICS")
    print(f"   Queries: {summary.passed_queries}/{summary.total_queries} passed")
    print(f"   Pass Rate: {summary.passed_queries/summary.total_queries*100:.1f}%")
    print(f"   Precision@{summary.limit}: {summary.mean_precision:.3f}")
    print(f"   Recall@{summary.limit}: {summary.mean_recall:.3f}")
    print(f"   MRR: {summary.mean_mrr:.3f}")
    
    print(f"\n📂 BY CATEGORY")
    for cat, metrics in sorted(summary.category_results.items()):
        status = "✅" if metrics['pass_rate'] >= 0.7 else "⚠️" if metrics['pass_rate'] >= 0.5 else "❌"
        print(f"   {status} {cat}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} MRR={metrics['mrr']:.2f} ({int(metrics['pass_rate']*100)}% pass)")
    
    print(f"\n📋 FAILED QUERIES")
    failed = [r for r in summary.query_results if not r.is_pass]
    if not failed:
        print("   None! All queries passed.")
    else:
        for r in failed[:10]:  # Show first 10
            print(f"   ❌ {r.query_id}: {r.question[:40]}...")
            print(f"      Found: {r.relevant_count}/{r.min_relevant} relevant")
            if r.retrieved_books:
                print(f"      Got: {', '.join(r.retrieved_books[:3])}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Alexandria Golden Set Evaluation')
    
    parser.add_argument('--golden-set', type=str, help='Path to golden_set.json')
    parser.add_argument('--limit', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold')
    parser.add_argument('--chunking-mode', type=str, choices=['semantic', 'fixed'],
                        help='Filter by chunking mode (A/B test)')
    parser.add_argument('--category', type=str, help='Filter by category')
    parser.add_argument('--format', type=str, choices=['text', 'json'], default='text',
                        help='Output format')
    parser.add_argument('--host', type=str, default=QDRANT_HOST, help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    
    args = parser.parse_args()
    
    summary = run_evaluation(
        golden_set_path=args.golden_set,
        limit=args.limit,
        threshold=args.threshold,
        host=args.host,
        port=args.port,
        chunking_mode=args.chunking_mode,
        category_filter=args.category
    )
    
    print_summary(summary, format=args.format)


if __name__ == '__main__':
    main()
