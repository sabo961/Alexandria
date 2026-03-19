#!/usr/bin/env python3
"""
Alexandria Reingest Mismatched Books
=====================================

Finds books where chunking_mode doesn't match current whitelist policy,
deletes their chunks from Qdrant, and reingests with correct mode.

Usage:
    # Dry run - see what would change
    python reingest_mismatched.py --dry-run

    # Actually reingest
    python reingest_mismatched.py
"""

import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import logging
from pathlib import Path

from config import (
    CALIBRE_LIBRARY_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    INGEST_LOG_DB,
)
from chunking_policy import load_whitelist, get_books_needing_reingest
from ingest_books import ingest_book, delete_book_from_qdrant
from calibre_db import CalibreDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_book_file(title: str, author: str, calibre_path: str) -> str:
    """Find the actual file path for a book in Calibre library."""
    try:
        calibre = CalibreDB(calibre_path)
        books = calibre.search_books(title)
        
        for book in books:
            if book.get('title', '').lower() == title.lower():
                # Found it - return first available format
                formats = book.get('formats', {})
                for fmt in ['.epub', '.pdf', '.txt', '.md']:
                    if fmt in formats:
                        return formats[fmt]
        
        return None
    except Exception as e:
        logger.warning(f"Could not find file for '{title}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Reingest books with wrong chunking mode')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--whitelist', type=str, help='Path to whitelist JSON')
    args = parser.parse_args()

    whitelist = load_whitelist(args.whitelist)
    logger.info(f"Loaded whitelist: {len(whitelist.get('authors', []))} authors")

    # Find mismatched books
    mismatched = get_books_needing_reingest(str(INGEST_LOG_DB), whitelist)
    
    if not mismatched:
        print("All books have correct chunking mode. Nothing to do.")
        return

    print(f"\nFound {len(mismatched)} books needing reingest:\n")
    print(f"{'Title':<50} {'Current':<10} {'Should Be':<10} {'Reason'}")
    print("-" * 90)
    
    for book in mismatched:
        title = book['title'][:47] + '...' if len(book['title']) > 50 else book['title']
        print(f"{title:<50} {book['current_mode']:<10} {book['should_be']:<10} {book['reason']}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would reingest {len(mismatched)} books.")
        return

    # Confirm
    confirm = input(f"\nReingest {len(mismatched)} books? [y/N] ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Reingest each book
    success = 0
    failed = 0
    
    for book in mismatched:
        title = book['title']
        author = book['author']
        should_semantic = book['should_be'] == 'semantic'
        
        print(f"\n[REINGEST] {title}")
        
        # Find file
        filepath = find_book_file(title, author, CALIBRE_LIBRARY_PATH)
        if not filepath:
            print(f"  [SKIP] Could not find file")
            failed += 1
            continue
        
        # Delete existing chunks
        print(f"  Deleting existing chunks...")
        try:
            delete_book_from_qdrant(
                title=title,
                collection_name='alexandria',
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT
            )
        except Exception as e:
            logger.warning(f"Delete failed (continuing anyway): {e}")
        
        # Reingest with correct mode
        print(f"  Reingesting with {'semantic' if should_semantic else 'fixed'} chunking...")
        try:
            result = ingest_book(
                filepath=filepath,
                collection_name='alexandria',
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT,
                hierarchical=True,
                force_reingest=True,
                use_semantic=should_semantic
            )
            
            if result.get('success'):
                print(f"  [OK] {result.get('chunks', 0)} chunks")
                success += 1
            else:
                print(f"  [FAIL] {result.get('error')}")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"REINGEST COMPLETE")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
