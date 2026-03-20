#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alexandria Chunking Policy
==========================

Determines whether a book should use semantic or fixed chunking.
Primary source: author_chunking SQLite table.
Fallback: JSON whitelist (deprecated).
"""

import json
import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Tuple

from config import ALEXANDRIA_DB

logger = logging.getLogger(__name__)

# Default whitelist path (check multiple locations)
_script_dir = Path(__file__).parent
_possible_paths = [
    _script_dir.parent / "config" / "semantic_whitelist.json",  # ../config/
    _script_dir / ".." / "config" / "semantic_whitelist.json",  # Same but different
    Path("C:/JedaiCloud/config/semantic_whitelist.json"),  # Windows absolute
    Path(os.environ.get("ALEXANDRIA_CONFIG", "")) / "semantic_whitelist.json",  # Env var
]
DEFAULT_WHITELIST = next((p for p in _possible_paths if p.exists()), _possible_paths[0])


def load_whitelist(whitelist_path: Optional[str] = None) -> dict:
    """Load chunking rules from JSON file. Supports both old and new format."""
    path = Path(whitelist_path) if whitelist_path else DEFAULT_WHITELIST
    
    # Also check for new chunking_rules.json
    rules_path = path.parent / "chunking_rules.json"
    if rules_path.exists():
        path = rules_path
    
    if not path.exists():
        logger.warning(f"Rules not found: {path}. Using empty rules.")
        return {"authors": [], "title_contains": [], "title_exact": []}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def get_author_mode_from_db(author_sort: str) -> Optional[str]:
    """
    Get chunking mode for author from SQLite database.
    
    Returns: 'semantic', 'fixed', 'none', or None if not found.
    """
    if not ALEXANDRIA_DB:
        return None
    
    try:
        conn = sqlite3.connect(ALEXANDRIA_DB)
        row = conn.execute(
            'SELECT mode FROM author_chunking WHERE author_sort = ?',
            (author_sort,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"DB lookup failed for '{author_sort}': {e}")
        return None


def get_title_mode_from_db(title: str) -> Optional[str]:
    """
    Check if title matches any pattern in title_chunking table.
    
    Returns: 'semantic', 'fixed', or None if no match.
    """
    if not ALEXANDRIA_DB or not title:
        return None
    
    try:
        conn = sqlite3.connect(ALEXANDRIA_DB)
        rows = conn.execute('SELECT pattern, mode FROM title_chunking').fetchall()
        conn.close()
        
        title_lower = title.lower()
        for pattern, mode in rows:
            if pattern.lower() in title_lower:
                return mode
        return None
    except Exception as e:
        logger.warning(f"Title DB lookup failed: {e}")
        return None


def get_book_chunking_mode(author_sort: str, title: str = "") -> Tuple[Optional[str], str]:
    """
    Determine chunking mode for a book.
    
    Priority:
    1. SQLite title_chunking patterns (highest - matches any title)
    2. SQLite author_chunking table
    3. JSON whitelist fallback (deprecated)
    
    Returns:
        (mode, reason) where mode is 'semantic', 'fixed', 'none', or None
    """
    # Check title patterns first (applies regardless of author)
    title_mode = get_title_mode_from_db(title)
    if title_mode:
        return title_mode, f"title_chunking pattern match"
    
    # Then check author
    mode = get_author_mode_from_db(author_sort)
    if mode:
        return mode, f"author_chunking DB: {author_sort}"
    
    # Fallback to JSON whitelist
    rules = load_whitelist()
    use_semantic, reason = should_use_semantic(title, author_sort, rules)
    
    if reason != "no rule match (default: fixed)":
        return 'semantic' if use_semantic else 'fixed', f"JSON whitelist: {reason}"
    
    # No match anywhere
    return None, "no match in DB or whitelist"


def should_use_semantic(
    title: str,
    author: str,
    rules: Optional[dict] = None
) -> Tuple[bool, str]:
    """
    Check if a book should use semantic chunking based on rules.
    
    Supports two formats:
    - Old: {"authors": ["Jung", ...]} — all semantic=true
    - New: {"authors": [{"pattern": "Jung", "semantic": true}, ...]}
    
    Returns:
        (use_semantic, reason)
    """
    if rules is None:
        rules = load_whitelist()
    
    title_lower = title.lower() if title else ""
    author_lower = author.lower() if author else ""
    
    # Check author patterns
    for item in rules.get("authors", []):
        # Support both old format (string) and new format (dict with semantic flag)
        if isinstance(item, str):
            pattern = item
            use_semantic = True
        else:
            pattern = item.get("pattern", "")
            use_semantic = item.get("semantic", True)
        
        if pattern.lower() in author_lower:
            return use_semantic, f"author matches '{pattern}'"
    
    # Check title contains
    for item in rules.get("title_contains", []):
        if isinstance(item, str):
            pattern = item
            use_semantic = True
        else:
            pattern = item.get("pattern", "")
            use_semantic = item.get("semantic", True)
        
        if pattern.lower() in title_lower:
            return use_semantic, f"title contains '{pattern}'"
    
    # Check exact title match
    for item in rules.get("title_exact", []):
        if isinstance(item, str):
            exact = item
            use_semantic = True
        else:
            exact = item.get("pattern", "")
            use_semantic = item.get("semantic", True)
        
        if exact.lower() == title_lower:
            return use_semantic, f"title exact match '{exact}'"
    
    return False, "no rule match (default: fixed)"


def get_books_needing_reingest(db_path: str, whitelist: Optional[dict] = None) -> list:
    """
    Find books where current chunking_mode doesn't match author_chunking policy.
    
    Uses SQLite author_chunking table as primary source.
    
    Returns list of dicts with title, author, current_mode, should_be
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    
    # Check if author_chunking table exists
    has_author_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='author_chunking'"
    ).fetchone()
    
    if not has_author_table:
        # Fallback to old JSON-based logic
        if whitelist is None:
            whitelist = load_whitelist()
        
        cursor = conn.execute(
            "SELECT DISTINCT book_title, author, chunking_mode FROM ingest_log WHERE success=1"
        )
        needs_reingest = []
        for row in cursor:
            title, author, current_mode = row
            current_mode = current_mode or 'fixed'
            should_semantic, reason = should_use_semantic(title, author, whitelist)
            should_mode = 'semantic' if should_semantic else 'fixed'
            if current_mode != should_mode:
                needs_reingest.append({
                    'title': title, 'author': author,
                    'current_mode': current_mode, 'should_be': should_mode, 'reason': reason
                })
        conn.close()
        return needs_reingest
    
    # Use SQLite author_chunking table (primary source)
    cursor = conn.execute("""
        SELECT il.book_title, il.author, il.chunking_mode, ac.mode
        FROM ingest_log il
        LEFT JOIN author_chunking ac ON il.author = ac.author_sort
        WHERE il.success = 1
          AND ac.mode IS NOT NULL
          AND ac.mode != 'none'
          AND (il.chunking_mode IS NULL OR il.chunking_mode != ac.mode)
    """)
    
    needs_reingest = []
    for row in cursor:
        title, author, current_mode, should_mode = row
        current_mode = current_mode or 'fixed'  # Old entries default to fixed
        needs_reingest.append({
            'title': title,
            'author': author,
            'current_mode': current_mode,
            'should_be': should_mode,
            'reason': f'author_chunking: {author}'
        })
    
    conn.close()
    return needs_reingest


if __name__ == "__main__":
    # Test mode
    import sys
    
    whitelist = load_whitelist()
    print(f"Loaded whitelist with {len(whitelist.get('authors', []))} authors")
    
    # Test cases
    tests = [
        ("Thus Spoke Zarathustra", "Friedrich Nietzsche"),
        ("Clean Code", "Robert C. Martin"),
        ("The Lord of the Rings", "J.R.R. Tolkien"),
        ("Software Architecture Patterns", "Mark Richards"),
        ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("Python Cookbook", "David Beazley"),
    ]
    
    print("\nTest results:")
    for title, author in tests:
        use_semantic, reason = should_use_semantic(title, author, whitelist)
        mode = "SEMANTIC" if use_semantic else "FIXED"
        print(f"  [{mode:8}] {title} by {author} — {reason}")
