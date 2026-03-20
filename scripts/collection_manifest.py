"""
Alexandria Collection Manifest (SQLite)

Tracks what's been ingested into each Qdrant collection.
Stored in shared SQLite database (ALEXANDRIA_DB) alongside ingest_log.

Usage:
    python collection_manifest.py show alexandria
    python collection_manifest.py list
    python collection_manifest.py sync alexandria
"""

import argparse
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_utils import check_qdrant_connection
from config import QDRANT_HOST, QDRANT_PORT, ALEXANDRIA_DB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fallback DB path if ALEXANDRIA_DB not configured
_LOCAL_FALLBACK_DB = str(Path(__file__).parent.parent / 'logs' / 'alexandria.db')


def _get_db_path() -> str:
    return ALEXANDRIA_DB if ALEXANDRIA_DB else _LOCAL_FALLBACK_DB


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection with schema ensured."""
    db_path = _get_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection TEXT NOT NULL,
        book_title TEXT NOT NULL,
        author TEXT DEFAULT 'Unknown',
        language TEXT DEFAULT 'unknown',
        source TEXT DEFAULT 'unknown',
        source_id TEXT DEFAULT '',
        file_path TEXT DEFAULT '',
        file_name TEXT DEFAULT '',
        file_type TEXT DEFAULT '',
        chunks_count INTEGER DEFAULT 0,
        file_size_mb REAL DEFAULT 0.0,
        ingested_at TEXT NOT NULL
    )''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_books_collection
                    ON books(collection)''')
    conn.execute('''CREATE INDEX IF NOT EXISTS idx_books_title
                    ON books(collection, book_title)''')
    return conn


class CollectionManifest:
    """Manage collection manifests in shared SQLite database."""

    def __init__(self, manifest_file: str = None, collection_name: Optional[str] = None):
        """
        Initialize CollectionManifest.

        Args:
            manifest_file: Ignored (kept for backward compatibility)
            collection_name: Collection name (used for scoped operations)
        """
        self.collection_name = collection_name

    def add_book(
        self,
        collection_name: str,
        book_path: str,
        book_title: str,
        author: str,
        chunks_count: int,
        file_size_mb: float,
        ingested_at: Optional[str] = None,
        file_type: Optional[str] = None,
        language: Optional[str] = None,
        source: Optional[str] = None,
        source_id: Optional[str] = None
    ):
        """Add book to manifest."""
        conn = _get_connection()

        # Check duplicate by source+source_id or by title
        if source and source_id:
            existing = conn.execute(
                'SELECT id FROM books WHERE collection=? AND source=? AND source_id=?',
                (collection_name, source, str(source_id))
            ).fetchone()
        else:
            existing = conn.execute(
                'SELECT id FROM books WHERE collection=? AND book_title=?',
                (collection_name, book_title)
            ).fetchone()

        if existing:
            logger.warning(f"Book already in manifest: {book_title} ({source}:{source_id})")
            conn.close()
            return

        if not file_type:
            file_type = Path(book_path).suffix.upper().replace('.', '')

        conn.execute(
            '''INSERT INTO books (collection, book_title, author, language,
               source, source_id, file_path, file_name, file_type,
               chunks_count, file_size_mb, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (collection_name, book_title, author or 'Unknown',
             language or 'unknown', source or 'unknown',
             str(source_id) if source_id else '',
             book_path, Path(book_path).name, file_type,
             chunks_count, round(file_size_mb, 2),
             ingested_at or datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        logger.info(f"Added to manifest: {book_title} ({chunks_count} chunks)")

    def is_ingested(self, collection_name: str, book_title: str, source: str = None, source_id: str = None) -> bool:
        """Check if a book is already tracked in the manifest."""
        conn = _get_connection()
        if source and source_id:
            row = conn.execute(
                'SELECT id FROM books WHERE collection=? AND source=? AND source_id=?',
                (collection_name, source, str(source_id))
            ).fetchone()
        else:
            row = conn.execute(
                'SELECT id FROM books WHERE collection=? AND book_title=?',
                (collection_name, book_title)
            ).fetchone()
        conn.close()
        return row is not None

    def remove_book(self, collection_name: str, book_title: str):
        """Remove book from manifest by title."""
        conn = _get_connection()
        cursor = conn.execute(
            'DELETE FROM books WHERE collection=? AND book_title=?',
            (collection_name, book_title)
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info(f"Removed from manifest: {book_title}")
        else:
            logger.error(f"Book not found: {book_title}")
        conn.close()

    def get_books(self, collection_name: str) -> List[Dict]:
        """Get all books for a collection."""
        conn = _get_connection()
        rows = conn.execute(
            'SELECT * FROM books WHERE collection=? ORDER BY ingested_at',
            (collection_name,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_summary(self, collection_name: str) -> Dict:
        """Get collection summary (total books, chunks, size)."""
        conn = _get_connection()
        row = conn.execute(
            '''SELECT COUNT(*) as book_count,
                      COALESCE(SUM(chunks_count), 0) as total_chunks,
                      COALESCE(SUM(file_size_mb), 0) as total_size_mb
               FROM books WHERE collection=?''',
            (collection_name,)
        ).fetchone()
        conn.close()
        return dict(row)

    def show_collection(self, collection_name: str):
        """Display collection contents."""
        summary = self.get_summary(collection_name)
        books = self.get_books(collection_name)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Collection: {collection_name}")
        logger.info("=" * 80)
        logger.info(f"Total books: {summary['book_count']}")
        logger.info(f"Total chunks: {summary['total_chunks']}")
        logger.info(f"Total size: {summary['total_size_mb']:.2f} MB")
        logger.info(f"Database: {_get_db_path()}")
        logger.info("")

        if not books:
            logger.info("No books in collection")
            return

        logger.info("Books:")
        logger.info("-" * 80)
        for idx, book in enumerate(books, 1):
            logger.info(f"\n{idx}. {book['book_title']}")
            logger.info(f"   Author: {book['author']}")
            logger.info(f"   Language: {book['language']}")
            logger.info(f"   Chunks: {book['chunks_count']}")
            logger.info(f"   Size: {book['file_size_mb']} MB")
            logger.info(f"   Source: {book['source']}:{book['source_id']}")
            logger.info(f"   Ingested: {book['ingested_at'][:10]}")
        logger.info("")
        logger.info("=" * 80)

    def list_collection_names(self) -> List[str]:
        """Return list of collection names that have books in manifest."""
        conn = _get_connection()
        rows = conn.execute(
            'SELECT DISTINCT collection FROM books ORDER BY collection'
        ).fetchall()
        conn.close()
        return [r['collection'] for r in rows]

    def list_collections(self):
        """List all collections."""
        conn = _get_connection()
        rows = conn.execute(
            '''SELECT collection,
                      COUNT(*) as book_count,
                      COALESCE(SUM(chunks_count), 0) as total_chunks,
                      COALESCE(SUM(file_size_mb), 0) as total_size_mb
               FROM books GROUP BY collection'''
        ).fetchall()
        conn.close()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Collection Manifest (SQLite)")
        logger.info("=" * 80)
        logger.info(f"Database: {_get_db_path()}")
        logger.info("")

        if not rows:
            logger.info("No collections in manifest")
            return

        for r in rows:
            logger.info(f"  {r['collection']}")
            logger.info(f"   Books: {r['book_count']}")
            logger.info(f"   Chunks: {r['total_chunks']}")
            logger.info(f"   Size: {r['total_size_mb']:.2f} MB")
            logger.info("")
        logger.info("=" * 80)

    def sync_with_qdrant(
        self,
        collection_name: str,
        host: str = None,
        port: int = None
    ):
        """Sync manifest with actual Qdrant collection (paginated scroll)."""
        host = host or QDRANT_HOST
        port = port or QDRANT_PORT

        logger.info(f"Syncing manifest with Qdrant: {collection_name}")

        is_connected, error_msg = check_qdrant_connection(host, port)
        if not is_connected:
            logger.error(error_msg)
            return

        client = QdrantClient(host=host, port=port)

        try:
            info = client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Collection not found: {e}")
            return

        total_points = info.points_count
        logger.info(f"Collection has {total_points} points. Scrolling...")

        # Paginated scroll
        qdrant_books = {}
        offset = None
        batch_size = 500

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            if not points:
                break

            for point in points:
                p = point.payload
                title = p.get('book_title', 'Unknown')
                if title not in qdrant_books:
                    qdrant_books[title] = {
                        'author': p.get('author', 'Unknown'),
                        'language': p.get('language', 'unknown'),
                        'source': p.get('source', 'unknown'),
                        'source_id': str(p.get('source_id', '')),
                        'chunks': 0
                    }
                qdrant_books[title]['chunks'] += 1

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Found {len(qdrant_books)} unique books in Qdrant")

        # Get existing manifest books
        existing_books = {b['book_title'] for b in self.get_books(collection_name)}

        # Add missing books
        added = 0
        for title, data in qdrant_books.items():
            if title not in existing_books:
                self.add_book(
                    collection_name=collection_name,
                    book_path='(synced from qdrant)',
                    book_title=title,
                    author=data['author'],
                    chunks_count=data['chunks'],
                    file_size_mb=0.0,
                    language=data['language'],
                    source=data['source'],
                    source_id=data['source_id'],
                    ingested_at='(synced)'
                )
                added += 1
            else:
                # Update chunk count
                conn = _get_connection()
                conn.execute(
                    'UPDATE books SET chunks_count=? WHERE collection=? AND book_title=?',
                    (data['chunks'], collection_name, title)
                )
                conn.commit()
                conn.close()

        # Show results
        logger.info("")
        for title, data in sorted(qdrant_books.items()):
            logger.info(f"  - {title} by {data['author']} [{data['language']}] ({data['chunks']} chunks)")

        if added > 0:
            logger.info(f"\nAdded {added} new books from Qdrant")
        else:
            logger.info(f"\nManifest already up to date")

    def verify_collection_exists(self, collection_name: str,
                                  qdrant_host: str = None,
                                  qdrant_port: int = None) -> bool:
        """Verify collection exists in Qdrant."""
        host = qdrant_host or QDRANT_HOST
        port = qdrant_port or QDRANT_PORT

        is_connected, error_msg = check_qdrant_connection(host, port)
        if not is_connected:
            return False

        try:
            client = QdrantClient(host=host, port=port)
            collections = [c.name for c in client.get_collections().collections]
            return collection_name in collections
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description='Alexandria Collection Manifest')
    parser.add_argument('command', choices=['show', 'list', 'sync', 'remove'],
                        help='Command to execute')
    parser.add_argument('collection', nargs='?', help='Collection name')
    parser.add_argument('--book', type=str, help='Book title (for remove)')
    parser.add_argument('--host', default=QDRANT_HOST)
    parser.add_argument('--port', type=int, default=QDRANT_PORT)

    args = parser.parse_args()
    manifest = CollectionManifest(collection_name=args.collection)

    if args.command == 'list':
        manifest.list_collections()
    elif args.command == 'show':
        if not args.collection:
            logger.error("Collection name required")
            return
        manifest.show_collection(args.collection)
    elif args.command == 'sync':
        if not args.collection:
            logger.error("Collection name required")
            return
        manifest.sync_with_qdrant(args.collection, args.host, args.port)
    elif args.command == 'remove':
        if not args.collection or not args.book:
            logger.error("Collection name and --book title required")
            return
        manifest.remove_book(args.collection, args.book)


if __name__ == '__main__':
    main()
