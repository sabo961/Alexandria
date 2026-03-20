#!/usr/bin/env python3
"""
Author Chunking Registry (SQLite)

Manages author → chunking mode mapping in ALEXANDRIA_DB.
Modes: semantic, fixed, none (skip)
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from config import ALEXANDRIA_DB, CALIBRE_LIBRARY_PATH

logger = logging.getLogger(__name__)

def _get_db_path() -> str:
    return ALEXANDRIA_DB if ALEXANDRIA_DB else str(Path(__file__).parent.parent / 'logs' / 'alexandria.db')

def _ensure_schema(conn: sqlite3.Connection):
    """Create author_chunking table if not exists."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS author_chunking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_sort TEXT UNIQUE NOT NULL,
            mode TEXT DEFAULT 'none' CHECK(mode IN ('semantic', 'fixed', 'none')),
            book_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_author_mode ON author_chunking(mode)')
    conn.commit()

def get_connection() -> sqlite3.Connection:
    """Get connection with schema ensured."""
    db_path = _get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn

def sync_from_calibre(calibre_db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Sync authors from Calibre database.
    Adds new authors with mode='none', preserves existing modes.
    
    Returns: {'added': N, 'existing': N, 'total': N}
    """
    calibre_path = calibre_db_path or str(Path(CALIBRE_LIBRARY_PATH) / 'metadata.db')
    
    # Connect to Calibre (read-only)
    calibre_conn = sqlite3.connect(f'file:{calibre_path}?mode=ro', uri=True)
    calibre_conn.row_factory = sqlite3.Row
    
    # Get all authors with book counts
    calibre_authors = calibre_conn.execute('''
        SELECT a.sort as author_sort, COUNT(DISTINCT b.id) as book_count
        FROM authors a
        LEFT JOIN books_authors_link bal ON a.id = bal.author
        LEFT JOIN books b ON bal.book = b.id
        GROUP BY a.sort
        ORDER BY a.sort
    ''').fetchall()
    calibre_conn.close()
    
    # Sync to our DB
    conn = get_connection()
    added = 0
    existing = 0
    now = datetime.now().isoformat()
    
    for row in calibre_authors:
        author = row['author_sort']
        book_count = row['book_count'] or 0
        
        # Check if exists
        exists = conn.execute(
            'SELECT id FROM author_chunking WHERE author_sort = ?', 
            (author,)
        ).fetchone()
        
        if exists:
            # Update book count only
            conn.execute(
                'UPDATE author_chunking SET book_count = ? WHERE author_sort = ?',
                (book_count, author)
            )
            existing += 1
        else:
            # Insert new with mode='none'
            conn.execute(
                'INSERT INTO author_chunking (author_sort, mode, book_count, updated_at) VALUES (?, ?, ?, ?)',
                (author, 'none', book_count, now)
            )
            added += 1
    
    conn.commit()
    conn.close()
    
    return {'added': added, 'existing': existing, 'total': added + existing}

def set_mode(author_sort: str, mode: str) -> bool:
    """Set chunking mode for an author."""
    if mode not in ('semantic', 'fixed', 'none'):
        raise ValueError(f"Invalid mode: {mode}")
    
    conn = get_connection()
    now = datetime.now().isoformat()
    cursor = conn.execute(
        'UPDATE author_chunking SET mode = ?, updated_at = ? WHERE author_sort = ?',
        (mode, now, author_sort)
    )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0

def bulk_set_mode(authors: List[str], mode: str) -> int:
    """Set mode for multiple authors. Returns count updated."""
    if mode not in ('semantic', 'fixed', 'none'):
        raise ValueError(f"Invalid mode: {mode}")
    
    conn = get_connection()
    now = datetime.now().isoformat()
    updated = 0
    
    for author in authors:
        cursor = conn.execute(
            'UPDATE author_chunking SET mode = ?, updated_at = ? WHERE author_sort = ?',
            (mode, now, author)
        )
        updated += cursor.rowcount
    
    conn.commit()
    conn.close()
    return updated

def get_mode(author_sort: str) -> Optional[str]:
    """Get chunking mode for an author."""
    conn = get_connection()
    row = conn.execute(
        'SELECT mode FROM author_chunking WHERE author_sort = ?',
        (author_sort,)
    ).fetchone()
    conn.close()
    return row['mode'] if row else None

def list_authors(
    mode: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict]:
    """List authors with optional filters."""
    conn = get_connection()
    
    query = 'SELECT author_sort, mode, book_count, updated_at FROM author_chunking WHERE 1=1'
    params = []
    
    if mode:
        query += ' AND mode = ?'
        params.append(mode)
    
    if search:
        query += ' AND author_sort LIKE ?'
        params.append(f'%{search}%')
    
    query += ' ORDER BY author_sort LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def count_authors(mode: Optional[str] = None, search: Optional[str] = None) -> int:
    """Count authors with optional filters."""
    conn = get_connection()
    
    query = 'SELECT COUNT(*) FROM author_chunking WHERE 1=1'
    params = []
    
    if mode:
        query += ' AND mode = ?'
        params.append(mode)
    
    if search:
        query += ' AND author_sort LIKE ?'
        params.append(f'%{search}%')
    
    count = conn.execute(query, params).fetchone()[0]
    conn.close()
    return count

def get_stats() -> Dict[str, int]:
    """Get counts by mode."""
    conn = get_connection()
    rows = conn.execute('''
        SELECT mode, COUNT(*) as count, SUM(book_count) as books
        FROM author_chunking
        GROUP BY mode
    ''').fetchall()
    conn.close()
    
    stats = {'semantic': 0, 'fixed': 0, 'none': 0, 'total_authors': 0, 'total_books': 0}
    for row in rows:
        stats[row['mode']] = row['count']
        stats['total_authors'] += row['count']
        stats['total_books'] += row['books'] or 0
    
    return stats

def import_from_json(json_path: str) -> int:
    """Import semantic authors from author_registry.json."""
    import json
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    authors_dict = data.get('authors', {})
    semantic_authors = [a for a, m in authors_dict.items() if m == 'semantic']
    
    return bulk_set_mode(semantic_authors, 'semantic')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Author Chunking Registry')
    parser.add_argument('command', choices=['sync', 'stats', 'list', 'import'])
    parser.add_argument('--mode', choices=['semantic', 'fixed', 'none'])
    parser.add_argument('--search', type=str)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--json', type=str, help='JSON file for import')
    
    args = parser.parse_args()
    
    if args.command == 'sync':
        result = sync_from_calibre()
        print(f"Synced: {result['added']} added, {result['existing']} existing, {result['total']} total")
    
    elif args.command == 'stats':
        stats = get_stats()
        print(f"Authors by mode:")
        print(f"  semantic: {stats['semantic']}")
        print(f"  fixed:    {stats['fixed']}")
        print(f"  none:     {stats['none']}")
        print(f"  TOTAL:    {stats['total_authors']} authors, {stats['total_books']} books")
    
    elif args.command == 'list':
        authors = list_authors(mode=args.mode, search=args.search, limit=args.limit)
        for a in authors:
            print(f"[{a['mode']:8}] {a['author_sort']} ({a['book_count']} books)")
    
    elif args.command == 'import':
        if not args.json:
            print("--json required for import")
        else:
            count = import_from_json(args.json)
            print(f"Imported {count} authors as semantic")
