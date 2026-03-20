#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexandria Batch Ingestion
===========================

Scan directory (Calibre library or custom folder) and ingest all books.

Usage:
    # Ingest from Calibre library (uses CALIBRE_LIBRARY_PATH from config)
    python batch_ingest.py

    # Ingest from custom directory
    python batch_ingest.py --directory "C:/My Books"

    # Dry run to see what would be ingested
    python batch_ingest.py --dry-run
"""

import sys
import os

# Force UTF-8 output on Windows (avoid encoding errors with Croatian/Czech names)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import argparse
import logging
import threading
import concurrent.futures
from functools import wraps
from pathlib import Path
from typing import List
import time

from config import (
    CALIBRE_LIBRARY_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    DEFAULT_EMBEDDING_MODEL,
)
from ingest_books import ingest_book
from chunking_policy import load_whitelist, should_use_semantic, get_book_chunking_mode
from collection_manifest import CollectionManifest
from calibre_db import CalibreDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timeout for single book processing (seconds)
# Large semantic books can legitimately take 30-60 minutes
# This is only for catching infinite hangs, not slow processing
BOOK_TIMEOUT = 3600  # 60 minutes

class BookTimeoutError(Exception):
    """Raised when book processing exceeds timeout."""
    pass

# Supported book formats
BOOK_FORMATS = {'.epub', '.pdf', '.txt', '.md', '.html', '.htm'}




def find_books(directory: str) -> List[Path]:
    """
    Recursively find all supported book files in directory.

    Args:
        directory: Root directory to scan

    Returns:
        List of Path objects for found books
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    books = []
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in BOOK_FORMATS:
            books.append(file_path)

    return sorted(books)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def batch_ingest(
    directory: str,
    collection_name: str = 'alexandria',
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    model_id: str = None,
    dry_run: bool = False
):
    """
    Batch ingest all books from directory.

    Args:
        directory: Directory to scan for books
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        model_id: Embedding model identifier
        dry_run: If True, only show what would be done
    """
    model_id = model_id or DEFAULT_EMBEDDING_MODEL

    print(f"\n{'='*70}")
    print(f"Alexandria Batch Ingestion")
    print(f"{'='*70}")
    print(f"Directory:   {directory}")
    print(f"Collection:  {collection_name}")
    print(f"Model:       {model_id}")
    print(f"Qdrant:      {qdrant_host}:{qdrant_port}")
    if dry_run:
        print(f"Mode:        DRY-RUN (no changes)")
    print(f"{'='*70}\n")

    # Find all books
    print("Scanning for books...")
    books = find_books(directory)

    if not books:
        print(f"[WARN] No books found in {directory}")
        return

    total = len(books)
    print(f"Found {total} book(s) to ingest\n")

    if dry_run:
        print("DRY-RUN - would ingest:")
        for i, book_path in enumerate(books, 1):
            print(f"  [{i}/{total}] {book_path.name}")
        print(f"\nTotal: {total} books")
        return

    # Ingest each book
    success_count = 0
    failed_count = 0
    failed_books = []
    start_time = time.time()
    book_times = []
    semantic_count = 0
    fixed_count = 0
    
    # Load manifest and Calibre DB once for the whole batch
    manifest = CollectionManifest(collection_name=collection_name)
    calibre_db = CalibreDB()
    whitelist = load_whitelist()  # Fallback only
    
    skipped_count = 0
    skipped_none = 0  # Authors with mode='none'

    for i, book_path in enumerate(books, 1):
        book_start = time.time()

        # Calculate ETA
        if book_times:
            avg_time = sum(book_times) / len(book_times)
            remaining = (total - i + 1) * avg_time
            eta_str = f" (ETA: {format_duration(remaining)})"
        else:
            eta_str = ""

        # Get real metadata from Calibre
        calibre_book = calibre_db.find_book_by_path(str(book_path))
        if calibre_book:
            title = calibre_book.title
            author_sort = calibre_book.author  # This is author_sort from Calibre
        else:
            # Fallback to filename parsing
            stem = book_path.stem
            if ' - ' in stem:
                parts = stem.rsplit(' - ', 1)
                title = parts[0]
                author_sort = parts[1] if len(parts) > 1 else ''
            else:
                title = stem
                author_sort = ''
        
        # Get chunking mode from SQLite (primary) or JSON (fallback)
        mode, reason = get_book_chunking_mode(author_sort, title)
        
        # Skip if mode is 'none' (not yet curated)
        if mode == 'none' or mode is None:
            # Don't spam log for every skipped book
            skipped_none += 1
            continue
        
        use_semantic = (mode == 'semantic')
        mode_str = mode.upper()

        # Skip if already tracked in manifest (prevents duplicate ingestion)
        if manifest.is_ingested(collection_name, title):
            print(f"\n[{i}/{total}] [SKIP] {book_path.name} (already in manifest)", flush=True)
            skipped_count += 1
            continue

        print(f"\n[{i}/{total}] [{mode_str}] {book_path.name}{eta_str}", flush=True)

        try:
            # Use ThreadPoolExecutor for timeout (works on Windows)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ingest_book,
                    filepath=str(book_path),
                    collection_name=collection_name,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port,
                    model_id=model_id,
                    hierarchical=True,
                    force_reingest=False,
                    use_semantic=use_semantic
                )
                try:
                    result = future.result(timeout=BOOK_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    raise BookTimeoutError(f"Timeout after {BOOK_TIMEOUT}s")

            if result.get('success'):
                book_duration = time.time() - book_start
                book_times.append(book_duration)
                chunks = result.get('chunks', 0)
                print(f"  [OK] {result['title']} - {chunks} chunks ({format_duration(book_duration)})", flush=True)
                success_count += 1
                if use_semantic:
                    semantic_count += 1
                else:
                    fixed_count += 1
            elif result.get('skipped'):
                # Bad metadata - skip without counting as failure
                error = result.get('error', 'Bad metadata')
                print(f"  [SKIP] {error}", flush=True)
                skipped_count += 1
            else:
                error = result.get('error', 'Unknown error')
                print(f"  [FAIL] {error}", flush=True)
                failed_count += 1
                failed_books.append({
                    'path': str(book_path),
                    'error': error
                })

        except BookTimeoutError as e:
            print(f"  [FAIL] TIMEOUT - {str(e)} - skipping", flush=True)
            logger.warning(f"TIMEOUT: {book_path.name} - {str(e)}")
            failed_count += 1
            failed_books.append({
                'path': str(book_path),
                'error': str(e)
            })
            # Log to separate timeout file for later review
            timeout_log = Path(directory) / '.qdrant' / 'timeout_books.txt'
            timeout_log.parent.mkdir(parents=True, exist_ok=True)
            with open(timeout_log, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {book_path.name} | {str(e)}\n")
                
        except Exception as e:
            print(f"  [FAIL] Exception: {str(e)}", flush=True)
            failed_count += 1
            failed_books.append({
                'path': str(book_path),
                'error': str(e)
            })

    # Print summary
    total_duration = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"BATCH INGESTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total books:   {total}")
    print(f"Not curated:   {skipped_none} (author mode='none', skipped)")
    print(f"Already done:  {skipped_count} (in manifest)")
    print(f"Succeeded:     {success_count} (fixed: {fixed_count}, semantic: {semantic_count})")
    print(f"Failed:        {failed_count}")
    print(f"Duration:      {format_duration(total_duration)}")

    if failed_books:
        print(f"\nFAILED BOOKS:")
        print(f"-" * 70)
        for failure in failed_books:
            print(f"  - {Path(failure['path']).name}")
            print(f"    Error: {failure['error']}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Batch ingest books from directory')
    parser.add_argument(
        '--directory', '-d',
        default=CALIBRE_LIBRARY_PATH,
        help=f'Directory to scan (default: {CALIBRE_LIBRARY_PATH})'
    )
    parser.add_argument(
        '--collection', '-c',
        default='alexandria',
        help='Qdrant collection name (default: alexandria)'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help=f'Embedding model (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    parser.add_argument(
        '--host',
        default=QDRANT_HOST,
        help=f'Qdrant host (default: {QDRANT_HOST})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=QDRANT_PORT,
        help=f'Qdrant port (default: {QDRANT_PORT})'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    try:
        batch_ingest(
            directory=args.directory,
            collection_name=args.collection,
            qdrant_host=args.host,
            qdrant_port=args.port,
            model_id=args.model,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
