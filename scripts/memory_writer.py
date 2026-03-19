#!/usr/bin/env python3
"""
Temenos Shared Memory Writer
=============================

Writes text into the shared memory SQLite database with semantic chunking,
FTS5 indexing, and vector embeddings. Zero external dependencies — stdlib only.

Used by both Claude Code and Rakušac (OpenClaw) to persist session memories
into a single searchable store.

Usage:
    python3 memory_writer.py --source claude-code --path "claude-code/2026-03-09.md" --text "..."
    python3 memory_writer.py --source claude-code --path "claude-code/2026-03-09.md" --file /path/to/file.md
    python3 memory_writer.py --search "keyword query"
    python3 memory_writer.py --list

Chunking: Alexandria Universal Semantic Chunker (block-based variant)
Embeddings: Ollama BGE-M3 (1024-dim, local)
Storage: SQLite with FTS5 + vec0
"""

import os
import sys
import json
import hashlib
import argparse
import sqlite3
import time
import re
import struct
import math
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional

# Config
SQLITE_PATH = Path(os.path.expanduser("~/.openclaw/memory/main.sqlite"))
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "bge-m3"
VECTOR_SIZE = 1024


# ---------------------------------------------------------------------------
# Math helpers (no numpy/sklearn needed)
# ---------------------------------------------------------------------------

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Ollama embedder (urllib, no requests)
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    def _embed_one(self, text: str) -> List[float]:
        data = json.dumps({"model": EMBEDDING_MODEL, "prompt": text}).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["embedding"]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def generate_embedding(self, text: str) -> List[float]:
        return self._embed_one(text)


# ---------------------------------------------------------------------------
# Universal Semantic Chunker (block-based, from qdrant-indexer.py)
# ---------------------------------------------------------------------------

class UniversalChunker:
    def __init__(
        self,
        embedding_model,
        threshold: float = 0.5,
        min_chunk_words: int = 50,
        max_chunk_words: int = 400
    ):
        self.model = embedding_model
        self.threshold = threshold
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words

    def _split_blocks(self, text: str) -> List[str]:
        blocks = re.split(r'\n\s*\n', text)
        return [b.strip() for b in blocks if b.strip() and len(b.strip()) > 2]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        blocks = self._split_blocks(text)
        if not blocks:
            return []

        if len(blocks) == 1:
            return [self._make_chunk(blocks[0], 0, metadata)]

        embeddings = self.model.generate_embeddings(blocks)

        chunks = []
        current_blocks = [blocks[0]]
        current_words = len(blocks[0].split())

        for i in range(1, len(blocks)):
            block = blocks[i]
            word_count = len(block.split())

            similarity = cosine_sim(embeddings[i - 1], embeddings[i])

            should_break = (similarity < self.threshold and current_words >= self.min_chunk_words)
            must_break = (current_words + word_count > self.max_chunk_words)

            if should_break or must_break:
                chunks.append(self._make_chunk("\n\n".join(current_blocks), len(chunks), metadata))
                current_blocks = [block]
                current_words = word_count
            else:
                current_blocks.append(block)
                current_words += word_count

        if current_blocks:
            chunks.append(self._make_chunk("\n\n".join(current_blocks), len(chunks), metadata))

        return chunks

    def _make_chunk(self, text: str, index: int, metadata: Optional[Dict]) -> Dict:
        chunk = {
            "text": text,
            "chunk_id": index,
            "word_count": len(text.split()),
            "strategy": "universal-semantic"
        }
        if metadata:
            chunk.update(metadata)
        return chunk


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def make_chunk_id(source: str, path: str, index: int) -> str:
    return hashlib.sha256(f"{source}:{path}:{index}".encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def serialize_vec0(embedding: List[float]) -> bytes:
    """Serialize embedding for sqlite-vec0 (little-endian float32 array)."""
    return struct.pack(f'<{len(embedding)}f', *embedding)


# ---------------------------------------------------------------------------
# Write memory
# ---------------------------------------------------------------------------

def write_memory(source: str, path: str, text: str):
    """Chunk, embed, and store text in the shared SQLite memory."""
    embedder = OllamaEmbedder()
    chunker = UniversalChunker(
        embedding_model=embedder,
        threshold=0.5,
        min_chunk_words=50,
        max_chunk_words=400
    )

    text_hash = content_hash(text)
    now = int(time.time())

    conn = get_db()

    # Check if unchanged
    existing = conn.execute(
        "SELECT hash FROM files WHERE path = ? AND source = ?",
        (path, source)
    ).fetchone()

    if existing and existing["hash"] == text_hash:
        print(f"  skip  {path} unchanged")
        conn.close()
        return

    # Delete old chunks for this path+source
    old_chunks = conn.execute(
        "SELECT id FROM chunks WHERE path = ? AND source = ?",
        (path, source)
    ).fetchall()

    for row in old_chunks:
        cid = row["id"]
        conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE id = ?", (cid,))
        try:
            conn.execute("DELETE FROM chunks_vec WHERE id = ?", (cid,))
        except Exception:
            pass

    # Chunk the text
    chunk_dicts = chunker.chunk(text)
    if not chunk_dicts:
        print(f"  warn  No chunks produced for {path}")
        conn.close()
        return

    # Generate embeddings and store
    for chunk_dict in chunk_dicts:
        chunk_text = chunk_dict["text"]
        idx = chunk_dict["chunk_id"]
        cid = make_chunk_id(source, path, idx)
        emb = embedder.generate_embedding(chunk_text)

        start_line = idx * 20 + 1
        end_line = start_line + len(chunk_text.split('\n'))

        # chunks table
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, path, source, start_line, end_line,
             content_hash(chunk_text), EMBEDDING_MODEL,
             chunk_text, json.dumps(emb), now)
        )

        # FTS5
        conn.execute(
            """INSERT OR REPLACE INTO chunks_fts
               (id, path, source, model, start_line, end_line, text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (cid, path, source, EMBEDDING_MODEL, start_line, end_line, chunk_text)
        )

        # vec0
        try:
            conn.execute(
                "INSERT OR REPLACE INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (cid, serialize_vec0(emb))
            )
        except Exception:
            pass  # vec0 extension may not be loaded

    # Update files table
    conn.execute(
        """INSERT OR REPLACE INTO files (path, source, hash, mtime, size)
           VALUES (?, ?, ?, ?, ?)""",
        (path, source, text_hash, now, len(text.encode()))
    )

    conn.commit()
    conn.close()
    print(f"  ok    {path} -> {len(chunk_dicts)} chunks (source: {source})")


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def search_fts(query: str, source: Optional[str] = None, limit: int = 5):
    """Full-text search across all memories."""
    # Escape FTS5 special characters for safe querying
    safe_query = '"' + query.replace('"', '""') + '"'
    conn = get_db()
    if source:
        rows = conn.execute(
            """SELECT c.path, c.source, c.text, c.updated_at
               FROM chunks c
               JOIN chunks_fts f ON c.id = f.id
               WHERE chunks_fts MATCH ? AND c.source = ?
               ORDER BY rank LIMIT ?""",
            (safe_query, source, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.path, c.source, c.text, c.updated_at
               FROM chunks c
               JOIN chunks_fts f ON c.id = f.id
               WHERE chunks_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (safe_query, limit)
        ).fetchall()
    conn.close()
    return rows


def list_sources():
    """List all indexed files grouped by source."""
    conn = get_db()
    rows = conn.execute(
        """SELECT source, path, size,
                  datetime(mtime, 'unixepoch', 'localtime') as modified
           FROM files ORDER BY source, path"""
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Temenos Shared Memory Writer")
    parser.add_argument("--source", default="claude-code", help="Source identifier (default: claude-code)")
    parser.add_argument("--path", help="Memory path (e.g., claude-code/2026-03-09.md)")
    parser.add_argument("--text", help="Text to store")
    parser.add_argument("--file", help="Read text from file instead of --text")
    parser.add_argument("--search", help="FTS5 search query")
    parser.add_argument("--search-source", help="Filter search by source")
    parser.add_argument("--list", action="store_true", help="List all indexed files")
    args = parser.parse_args()

    if args.search:
        results = search_fts(args.search, source=args.search_source)
        if not results:
            print("No results found.")
            return
        for r in results:
            print(f"\n--- [{r['source']}] {r['path']} ---")
            print(r["text"][:300])
        return

    if args.list:
        rows = list_sources()
        current_source = None
        for r in rows:
            if r["source"] != current_source:
                current_source = r["source"]
                print(f"\n[{current_source}]")
            print(f"  {r['path']}  ({r['size']} bytes, {r['modified']})")
        return

    if not args.path:
        parser.error("--path is required for writing")

    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Empty text, nothing to write.")
        return

    write_memory(args.source, args.path, text)


if __name__ == "__main__":
    main()
