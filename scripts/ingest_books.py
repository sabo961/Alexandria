"""
Alexandria Book Ingestion Script

Unified ingestion pipeline for EPUB, PDF, and TXT files.
Utilizes Universal Semantic Chunking for high-quality retrieval.

Usage:
    python ingest_books.py --file "path/to/book.epub" --collection alexandria
    python ingest_books.py --file "path/to/book.epub" --dry-run --threshold 0.55
    python ingest_books.py --file "path/to/book.epub" --compare
"""

import os
# Disable tqdm globally to avoid stderr issues in Streamlit
os.environ['TQDM_DISABLE'] = '1'

import argparse
import uuid
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from calibre_db import CalibreDB # Import CalibreDB

# Book parsing libraries
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# NLP & Embeddings
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_utils import check_qdrant_connection

# Chunking options
from universal_chunking import UniversalChunker
from fixed_chunking import FixedChunker

# Hierarchical Chunking
from chapter_detection import detect_chapters

# Central configuration
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    CALIBRE_LIBRARY_PATH,
    EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    INGEST_VERSION,
    ALEXANDRIA_DB,
)

# Collection manifest tracking
from collection_manifest import CollectionManifest

# GPU Optimization: TF32 for faster matmul on Ampere+ GPUs
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# Setup logging - configurable via ALEXANDRIA_LOG_LEVEL environment variable
# Usage: export ALEXANDRIA_LOG_LEVEL=DEBUG or ALEXANDRIA_LOG_LEVEL=INFO
log_level_str = os.getenv('ALEXANDRIA_LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# INGEST PERFORMANCE LOG (SQLite)
# ============================================================================

_LOCAL_FALLBACK_DB = Path(__file__).parent.parent / 'logs' / 'alexandria.db'
INGEST_LOG_DB = Path(ALEXANDRIA_DB) if ALEXANDRIA_DB else _LOCAL_FALLBACK_DB
_HOSTNAME = os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))

def _ensure_ingest_log_schema(conn):
    """Create/migrate ingest_log table schema."""
    conn.execute('''CREATE TABLE IF NOT EXISTS ingest_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        hostname TEXT,
        book_title TEXT,
        author TEXT,
        language TEXT,
        source TEXT,
        source_id TEXT,
        collection TEXT,
        chunks INTEGER,
        file_size_mb REAL,
        duration_total REAL,
        duration_embed REAL,
        duration_chunk REAL,
        duration_upload REAL,
        chunks_per_sec REAL,
        device TEXT,
        model_id TEXT,
        batch_size INTEGER,
        success INTEGER
    )''')
    # Migration: add hostname column if missing (for existing DBs)
    try:
        conn.execute('SELECT hostname FROM ingest_log LIMIT 1')
    except sqlite3.OperationalError:
        conn.execute('ALTER TABLE ingest_log ADD COLUMN hostname TEXT')
    # Migration: add chunking_mode column if missing
    try:
        conn.execute('SELECT chunking_mode FROM ingest_log LIMIT 1')
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE ingest_log ADD COLUMN chunking_mode TEXT DEFAULT 'semantic'")

def _log_ingest_performance(book_title: str, author: str, language: str,
                            chunks: int, file_size_mb: float,
                            duration_total: float, duration_embed: float,
                            duration_chunk: float, duration_upload: float,
                            device: str, model_id: str, batch_size: int,
                            collection: str, success: bool,
                            source: str = '', source_id: str = '',
                            chunking_mode: str = 'fixed'):
    """Log ingest job to SQLite for collection-level performance tracking."""
    try:
        INGEST_LOG_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(INGEST_LOG_DB))
        _ensure_ingest_log_schema(conn)
        chunks_per_sec = chunks / duration_embed if duration_embed > 0 else 0
        conn.execute(
            '''INSERT INTO ingest_log (timestamp, hostname, book_title, author,
               language, source, source_id, collection, chunks, file_size_mb,
               duration_total, duration_embed, duration_chunk, duration_upload,
               chunks_per_sec, device, model_id, batch_size, success, chunking_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (datetime.now().isoformat(), _HOSTNAME, book_title, author,
             language, source, source_id, collection, chunks, round(file_size_mb, 3),
             round(duration_total, 2), round(duration_embed, 2),
             round(duration_chunk, 2), round(duration_upload, 2),
             round(chunks_per_sec, 2), device, model_id, batch_size,
             1 if success else 0, chunking_mode)
        )
        conn.commit()
        conn.close()
        logger.info(f"Performance logged: {chunks} chunks in {duration_total:.1f}s "
                    f"({chunks_per_sec:.1f} chunks/sec, device={device}, host={_HOSTNAME})")
    except Exception as e:
        logger.warning(f"Failed to log performance (non-critical): {e}")

# Lazy load Calibre DB (uses CALIBRE_LIBRARY_PATH from config)
calibre_db_instance = None


# ============================================================================ 
# TEXT EXTRACTION
# ============================================================================ 

def normalize_file_path(filepath: str) -> Tuple[str, str, bool, int]:
    """Normalize paths for cross-platform file access."""
    logger.debug(f"normalize_file_path input: {repr(filepath)}")
    expanded = os.path.expanduser(filepath)
    logger.debug(f"After expanduser: {repr(expanded)}")
    abs_path = os.path.abspath(expanded)
    logger.debug(f"After abspath: {repr(abs_path)}")
    path_for_open = abs_path
    used_long_path = False

    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        if len(abs_path) >= 248:
            if abs_path.startswith("\\\\"):
                path_for_open = "\\\\?\\UNC\\" + abs_path.lstrip("\\")
            else:
                path_for_open = "\\\\?\\" + abs_path
            used_long_path = True
            logger.debug(f"Applied long path prefix: {repr(path_for_open)}")

    logger.debug(f"normalize_file_path output: path_for_open={repr(path_for_open)}, abs_path={repr(abs_path)}")
    return path_for_open, abs_path, used_long_path, len(abs_path)


def validate_file_access(path_for_open: str, display_path: str) -> Tuple[bool, Optional[str]]:
    """Validate file exists and is readable."""
    # NOTE: Cannot use sys.stderr in Streamlit - it causes [Errno 22]
    logger.debug(f"validate_file_access checking: {repr(path_for_open)}")

    if not os.path.exists(path_for_open):
        logger.debug(f"File does not exist: {path_for_open}")
        return False, f"File not found: {display_path}"

    try:
        logger.debug(f"Calling os.stat()...")
        os.stat(path_for_open)
        logger.debug(f"os.stat() succeeded, attempting to open file...")
        with open(path_for_open, 'rb') as f:
            f.read(1)
        logger.debug(f"File opened and read successfully")
        return True, None
    except OSError as e:
        logger.error(f"OSError during file access: {e.__class__.__name__}: {e}", exc_info=True)
        return False, f"{e.__class__.__name__}: {e}"


def extract_text(filepath: str) -> Tuple[str, Dict]:
    """
    Extract text from file based on extension.
    Returns: full_text, metadata
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == '.epub':
        book = epub.read_epub(filepath)
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                if text: chapters.append(text)
        
        metadata = {
            'title': _get_epub_metadata(book, 'title'),
            'author': _get_epub_metadata(book, 'creator'),
            'language': _get_epub_metadata(book, 'language'),
            'format': 'EPUB'
        }
        return "\n\n".join(chapters), metadata

    elif ext == '.pdf':
        doc = fitz.open(filepath)
        pages = [page.get_text() for page in doc]
        metadata = {
            'title': doc.metadata.get('title', 'Unknown'),
            'author': doc.metadata.get('author', 'Unknown'),
            'language': standardize_language_code(doc.metadata.get('language') or 'unknown'),
            'format': 'PDF'
        }
        doc.close()
        return "\n\n".join(pages), metadata

    elif ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        fmt = 'MD' if ext == '.md' else 'TXT'
        return text, {'title': Path(filepath).stem, 'author': 'Unknown', 'format': fmt}

    elif ext in ['.html', '.htm']:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        soup = BeautifulSoup(content, 'html.parser')
        # Try to get title from <title> tag
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else Path(filepath).stem
        # Try to get author from meta tag
        author_meta = soup.find('meta', attrs={'name': 'author'})
        author = author_meta.get('content', 'Unknown') if author_meta else 'Unknown'
        # Extract text content
        text = soup.get_text(separator='\n', strip=True)
        return text, {'title': title, 'author': author, 'format': 'HTML'}

    else:
        raise ValueError(f"Unsupported format: {ext}")


def _get_epub_metadata(book, key: str) -> str:
    try:
        result = book.get_metadata('DC', key)
        if result: 
            return standardize_language_code(result[0][0])
    except: pass
    return "unknown"

def standardize_language_code(lang: str) -> str:
    """Standardizes common language codes to a consistent format (e.g., 'en', 'hr')."""
    if not lang:
        return "unknown"
    
    lang = lang.lower().strip()

    # Standardize known problematic English indicators
    if lang == 'eng': # Only 'eng' to 'en', keep 'en-us', 'en-gb'
        return 'en'
    
    # Standardize known problematic Croatian indicators
    if lang in ['hrv', 'cro']:
        return 'hr' 
    
    # Keep other specific regional variants or unexpected codes as they are
    return lang


# ============================================================================
# EMBEDDINGS
# ============================================================================

class EmbeddingGenerator:
    """Singleton with multi-model cache."""

    _instance = None
    _models = {}  # Cache per model_id

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ollama_embed(self, texts: List[str], model_id: str) -> List[List[float]]:
        """Generate embeddings via Ollama API (GPU-accelerated, no download needed)."""
        import urllib.request
        import json as _json

        ollama_model = EMBEDDING_MODELS[model_id]["name"].split("/")[-1]  # BAAI/bge-m3 → bge-m3
        # Use ollama_model name as registered in Ollama (bge-m3:latest)
        ollama_name = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3:latest")
        ollama_url = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434") + "/api/embeddings"

        results = []
        for text in texts:
            # Skip empty/whitespace-only chunks
            clean = text.strip()
            if not clean:
                results.append(None)  # Mark for filtering
                continue
            payload = _json.dumps({"model": ollama_name, "prompt": clean}).encode()
            req = urllib.request.Request(ollama_url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = _json.loads(resp.read())
                emb = data["embedding"]
                # Check for NaN values (Ollama bug on certain inputs)
                has_nan = any((isinstance(v, float) and v != v) for v in emb)
                if has_nan:
                    logger.warning(f"Ollama returned NaN for chunk (len={len(clean)}) — skipping")
                    results.append(None)
                else:
                    results.append(emb)
            except Exception as e:
                logger.warning(f"Ollama embed failed for chunk (len={len(clean)}): {e} — skipping")
                results.append(None)
        return results

    def get_model(self, model_id: str = None):
        """
        Get or load embedding model with GPU/CPU detection.
        Returns None when using Ollama backend (no local model needed).
        """
        model_id = model_id or DEFAULT_EMBEDDING_MODEL
        # Ollama backend: no local model to load
        if os.environ.get("EMBEDDING_BACKEND") == "ollama":
            return None
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed and EMBEDDING_BACKEND != ollama")

        import torch

        if model_id not in self._models:
            if model_id not in EMBEDDING_MODELS:
                raise ValueError(
                    f"Unknown model_id: {model_id}. Available: {list(EMBEDDING_MODELS.keys())}"
                )

            model_config = EMBEDDING_MODELS[model_id]
            model_name = model_config["name"]
            expected_dim = model_config["dim"]

            if EMBEDDING_DEVICE == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = EMBEDDING_DEVICE

            if device == 'cpu':
                logger.warning("Running on CPU - embedding generation will be slower")

            logger.info(f"Loading embedding model: {model_name} (id: {model_id})")
            logger.info(f"Device: {device}")

            model = SentenceTransformer(model_name, device=device)
            actual_dim = model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {actual_dim}")
            if actual_dim != expected_dim:
                logger.warning(f"Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            self._models[model_id] = model

        return self._models[model_id]

    def get_model_config(self, model_id: str = None) -> dict:
        """Get model configuration (name, dim) without loading the model."""
        model_id = model_id or DEFAULT_EMBEDDING_MODEL
        return EMBEDDING_MODELS.get(model_id)

    def generate_embeddings(self, texts: List[str], model_id: str = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Uses Ollama API if EMBEDDING_BACKEND=ollama, otherwise sentence-transformers.
        """
        model_id = model_id or DEFAULT_EMBEDDING_MODEL

        # Ollama backend (GPU via existing Ollama service)
        if os.environ.get("EMBEDDING_BACKEND") == "ollama":
            logger.info(f"Embedding via Ollama ({len(texts)} texts)")
            return self._ollama_embed(texts, model_id)

        # sentence-transformers backend
        import torch
        model = self.get_model(model_id)
        batch_size = 64 if model.device.type == 'cuda' else 32

        if model.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
        else:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )

        logger.debug(f"Generated {len(texts)} embeddings of dimension {embeddings.shape[1]}")
        return embeddings.tolist()

def generate_embeddings(texts: List[str], model_id: str = None) -> List[List[float]]:
    return EmbeddingGenerator().generate_embeddings(texts, model_id)


# ============================================================================ 
# QDRANT UPLOAD
# ============================================================================ 

def upload_to_qdrant(
    chunks: List[Dict],
    embeddings: List[List[float]],
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    model_id: Optional[str] = None
) -> Dict:
    """
    Upload chunks to Qdrant vector database.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        embeddings: List of embedding vectors
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        model_id: Embedding model identifier (default: DEFAULT_EMBEDDING_MODEL)

    Returns:
        Dict with 'success' (bool) and 'error' (str) if failed
    """
    if not chunks:
        return {'success': True, 'uploaded': 0}

    # Get model configuration for metadata
    model_id = model_id or DEFAULT_EMBEDDING_MODEL
    if model_id not in EMBEDDING_MODELS:
        error_msg = f"Unknown embedding model: '{model_id}'. Available: {list(EMBEDDING_MODELS.keys())}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}
    model_config = EMBEDDING_MODELS[model_id]

    # Check connection first
    is_connected, error_msg = check_qdrant_connection(qdrant_host, qdrant_port)
    if not is_connected:
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}

    try:
        # Wrap QdrantClient instantiation
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    except Exception as e:
        error_detail = f"""
[ERROR] Cannot instantiate Qdrant client at {qdrant_host}:{qdrant_port}

Possible causes:
  1. VPN not connected - Verify VPN connection if server is remote
  2. Firewall blocking port {qdrant_port} - Check firewall rules
  3. Qdrant server not running - Verify server status at http://{qdrant_host}:{qdrant_port}/dashboard
  4. Network issue - Server may be slow or unreachable

Connection error: {str(e)}
"""
        logger.error(f"QdrantClient instantiation failed: {str(e)}")
        return {'success': False, 'error': error_detail.strip()}

    try:
        # Wrap collection operations
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )
    except Exception as e:
        error_detail = f"""
[ERROR] Failed to check/create collection '{collection_name}' at {qdrant_host}:{qdrant_port}

Possible causes:
  1. Network connection lost mid-operation - Check network stability
  2. Insufficient permissions - Verify Qdrant server permissions
  3. Server overloaded - Check Qdrant server resources
  4. Invalid collection configuration - Verify vector dimensions

Collection error: {str(e)}
"""
        logger.error(f"Qdrant collection operation failed: {str(e)}")
        return {'success': False, 'error': error_detail.strip()}

    # Build points (skip chunks where embedding is None — failed Ollama calls)
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            continue
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk['text'],
                "book_title": chunk.get('title', chunk.get('book_title', 'Unknown')),
                "author": chunk.get('author', 'Unknown'),
                "section_name": chunk.get('section_name', ''),
                "language": chunk.get('language', 'unknown'),
                "ingested_at": datetime.now().isoformat(),
                "strategy": "universal-semantic",
                # Embedding model metadata for query auto-detection
                "embedding_model_id": model_id,
                "embedding_model_name": model_config.get("name", "unknown"),
                "embedding_dimension": model_config.get("dim", len(embedding)),
                "ingest_version": INGEST_VERSION
            }
        ))

    try:
        # Wrap batch upload operations
        for i in range(0, len(points), 100):
            client.upsert(collection_name=collection_name, points=points[i:i+100])
    except Exception as e:
        error_detail = f"""
[ERROR] Failed to upload points to '{collection_name}' at {qdrant_host}:{qdrant_port}

Possible causes:
  1. Network connection lost during upload - Check network stability
  2. Server timeout - Large batch may exceed timeout limits
  3. Insufficient disk space on Qdrant server - Check server storage
  4. Server crashed mid-operation - Verify server status

Upload error: {str(e)}
"""
        logger.error(f"Qdrant upsert operation failed: {str(e)}")
        return {'success': False, 'error': error_detail.strip()}

    logger.info(f"[OK] Uploaded {len(points)} semantic chunks to '{collection_name}'")
    return {'success': True, 'uploaded': len(points)}


def upload_hierarchical_to_qdrant(
    parent_chunks: List[Dict],
    parent_embeddings: List[List[float]],
    child_chunks: List[Dict],
    child_embeddings: List[List[float]],
    collection_name: str,
    qdrant_host: str,
    qdrant_port: int,
    model_id: Optional[str] = None
) -> Dict:
    """
    Upload hierarchical chunks (parents + children) to Qdrant.

    Parents are uploaded first, then children with parent_id references.

    Args:
        parent_chunks: List of parent (chapter) chunk dictionaries
        parent_embeddings: List of parent embedding vectors
        child_chunks: List of child (semantic) chunk dictionaries
        child_embeddings: List of child embedding vectors
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        model_id: Embedding model identifier (default: DEFAULT_EMBEDDING_MODEL)

    Returns:
        Dict with 'success', 'parent_count', 'child_count', 'error'
    """
    if not parent_chunks and not child_chunks:
        return {'success': True, 'parent_count': 0, 'child_count': 0}

    # Get model configuration for metadata
    model_id = model_id or DEFAULT_EMBEDDING_MODEL
    if model_id not in EMBEDDING_MODELS:
        error_msg = f"Unknown embedding model: '{model_id}'. Available: {list(EMBEDDING_MODELS.keys())}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}
    model_config = EMBEDDING_MODELS[model_id]

    # Check connection
    is_connected, error_msg = check_qdrant_connection(qdrant_host, qdrant_port)
    if not is_connected:
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
    except Exception as e:
        logger.error(f"QdrantClient instantiation failed: {str(e)}")
        return {'success': False, 'error': str(e)}

    # Ensure collection exists
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            # Use parent embedding size (should be same as child)
            vector_size = len(parent_embeddings[0]) if parent_embeddings else len(child_embeddings[0])
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Collection operation failed: {str(e)}")
        return {'success': False, 'error': str(e)}

    # Build parent points (skip None embeddings)
    parent_points = []
    for chunk, embedding in zip(parent_chunks, parent_embeddings):
        if embedding is None:
            continue
        parent_points.append(PointStruct(
            id=chunk['id'],  # Use pre-assigned UUID
            vector=embedding,
            payload={
                "text": chunk.get('text', ''),
                "full_text": chunk.get('full_text', chunk.get('text', '')),
                "book_title": chunk.get('book_title', 'Unknown'),
                "author": chunk.get('author', 'Unknown'),
                "section_name": chunk.get('section_name', ''),
                "section_index": chunk.get('section_index', 0),
                "language": chunk.get('language', 'unknown'),
                "source": chunk.get('source', 'unknown'),
                "source_id": chunk.get('source_id', ''),
                "chunk_level": "parent",
                "child_count": chunk.get('child_count', 0),
                "token_count": chunk.get('token_count', 0),
                "ingested_at": datetime.now().isoformat(),
                "strategy": "hierarchical",
                # Embedding model metadata for query auto-detection
                "embedding_model_id": model_id,
                "embedding_model_name": model_config.get("name", "unknown"),
                "embedding_dimension": model_config.get("dim", len(embedding)),
                "ingest_version": INGEST_VERSION
            }
        ))

    # Build child points (skip None embeddings)
    child_points = []
    for chunk, embedding in zip(child_chunks, child_embeddings):
        if embedding is None:
            continue
        child_points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.get('text', ''),
                "book_title": chunk.get('book_title', 'Unknown'),
                "author": chunk.get('author', 'Unknown'),
                "section_name": chunk.get('section_name', ''),
                "language": chunk.get('language', 'unknown'),
                "source": chunk.get('source', 'unknown'),
                "source_id": chunk.get('source_id', ''),
                "chunk_level": "child",
                "parent_id": chunk.get('parent_id', ''),
                "sequence_index": chunk.get('sequence_index', 0),
                "sibling_count": chunk.get('sibling_count', 0),
                "token_count": chunk.get('token_count', len(chunk.get('text', '').split())),
                "ingested_at": datetime.now().isoformat(),
                "strategy": "universal-semantic",
                # Embedding model metadata for query auto-detection
                "embedding_model_id": model_id,
                "embedding_model_name": model_config.get("name", "unknown"),
                "embedding_dimension": model_config.get("dim", len(embedding)),
                "ingest_version": INGEST_VERSION
            }
        ))

    # Upload parents first
    try:
        for i in range(0, len(parent_points), 100):
            client.upsert(collection_name=collection_name, points=parent_points[i:i+100])
        logger.info(f"[OK] Uploaded {len(parent_points)} parent chunks")
    except Exception as e:
        logger.error(f"Parent upload failed: {str(e)}")
        return {'success': False, 'error': f"Parent upload failed: {str(e)}"}

    # Upload children
    try:
        for i in range(0, len(child_points), 100):
            client.upsert(collection_name=collection_name, points=child_points[i:i+100])
        logger.info(f"[OK] Uploaded {len(child_points)} child chunks")
    except Exception as e:
        logger.error(f"Child upload failed: {str(e)}")
        return {'success': False, 'error': f"Child upload failed: {str(e)}"}

    logger.info(f"[OK] Hierarchical upload complete: {len(parent_points)} parents, {len(child_points)} children")
    return {
        'success': True,
        'parent_count': len(parent_points),
        'child_count': len(child_points),
        'uploaded': len(parent_points) + len(child_points)
    }


def truncate_for_embedding(text: str, max_tokens: int = 8192) -> str:
    """Truncate text to fit embedding model's token limit."""
    # Rough estimate: 1.3 tokens per word
    max_words = int(max_tokens / 1.3)
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + ' [truncated]'


def delete_book_chunks(
    book_title: str,
    collection_name: str,
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT
) -> int:
    """
    Delete all chunks for a book from Qdrant.

    Args:
        book_title: Title of the book to delete chunks for
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port

    Returns:
        Number of chunks deleted
    """
    # Check connection first
    is_connected, error_msg = check_qdrant_connection(qdrant_host, qdrant_port)
    if not is_connected:
        logger.error(error_msg)
        return 0

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            logger.warning(f"Collection '{collection_name}' not found - nothing to delete")
            return 0

        # Count existing points for this book before deletion
        count_result = client.count(
            collection_name=collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(key="book_title", match=MatchValue(value=book_title))
                ]
            )
        )
        existing_count = count_result.count

        if existing_count == 0:
            logger.info(f"No chunks found for '{book_title}' in '{collection_name}'")
            return 0

        # Delete all points matching the book_title
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="book_title", match=MatchValue(value=book_title))
                ]
            )
        )

        logger.info(f"Deleted {existing_count} chunks for '{book_title}' from '{collection_name}'")
        return existing_count

    except Exception as e:
        logger.error(f"Failed to delete chunks for '{book_title}': {e}")
        return 0


def _get_calibre_db() -> Optional[CalibreDB]:
    """Lazy-loads and returns a CalibreDB instance."""
    global calibre_db_instance
    if calibre_db_instance is None:
        try:
            calibre_db_instance = CalibreDB(CALIBRE_LIBRARY_PATH)
        except FileNotFoundError:
            logger.warning(f"Calibre DB not found at {CALIBRE_LIBRARY_PATH}. Metadata enrichment from Calibre will be skipped.")
            calibre_db_instance = None
        except Exception as e:
            logger.error(f"Failed to connect to Calibre DB at {CALIBRE_LIBRARY_PATH}: {e}. Metadata enrichment from Calibre will be skipped.")
            calibre_db_instance = None
    return calibre_db_instance

def _enrich_metadata_from_calibre(filepath: str, metadata: Dict) -> Dict:
    """
    Attempts to enrich metadata from Calibre DB if a match is found.
    Prioritizes Calibre data for title, author, and language if current metadata is 'Unknown' or 'unknown'.
    """
    db = _get_calibre_db()
    if not db:
        return metadata # Cannot enrich without Calibre DB

    filename = Path(filepath).name
    calibre_book = db.match_file_to_book(filename)

    if calibre_book:
        logger.info(f"Matched '{filename}' to Calibre book: '{calibre_book.title}' by '{calibre_book.author}'.")
        # Override 'Unknown' or 'unknown' fields with Calibre data
        if metadata.get('title', 'Unknown') in ['Unknown', 'unknown']:
            metadata['title'] = calibre_book.title
        # Calibre stores authors as "Author1 & Author2", but metadata expects a string.
        # CalibreBook.author field already handles this.
        if metadata.get('author', 'Unknown') in ['Unknown', 'unknown']:
            metadata['author'] = calibre_book.author

        if metadata.get('language', 'unknown') == 'unknown':
            # Use standardized language from Calibre if available and current is 'unknown'
            metadata['language'] = standardize_language_code(calibre_book.language)
        
        # Also ensure format is consistent (take first format from Calibre and uppercase it)
        if calibre_book.formats and metadata.get('format', 'unknown') == 'unknown':
             metadata['format'] = calibre_book.formats[0].upper()

    return metadata





def extract_metadata_only(filepath: str) -> Dict:
    """
    Extracts only metadata (title, author, language) from a file without processing content.
    Returns: Dict with 'title', 'author', 'language', 'format', or 'error' if unsupported/failed.
    """
    normalized_path, display_path, _, _ = normalize_file_path(filepath)

    ok, err = validate_file_access(normalized_path, display_path)
    if not ok: 
        return {'error': err, 'filepath': display_path}

    try:
        _, metadata = extract_text(normalized_path)
        metadata['filepath'] = display_path
        
        # Enrich metadata from Calibre if available
        metadata = _enrich_metadata_from_calibre(filepath, metadata)

        return metadata
    except ValueError as e:
        return {'error': str(e), 'filepath': display_path}
    except Exception as e:
        logger.error(f"Error extracting metadata from {display_path}: {e}")
        return {'error': f"Failed to extract metadata: {e}", 'filepath': display_path}


# ============================================================================ 
# MAIN PIPELINE
# ============================================================================ 

def ingest_book(
    filepath: str,
    collection_name: str = 'alexandria',
    qdrant_host: str = 'localhost',
    qdrant_port: int = 6333,
    language_override: Optional[str] = None,
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    hierarchical: bool = True,
    threshold: float = 0.55,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1200,
    force_reingest: bool = False,
    model_id: Optional[str] = None,
    source_meta: Optional[Dict] = None,
    use_semantic: bool = False  # False = fixed chunking (fast), True = semantic chunking (slow)
):
    """
    Ingest a book into Qdrant with optional hierarchical chunking.

    Args:
        filepath: Path to book file (EPUB, PDF, TXT, MD, HTML)
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        language_override: Override detected language
        title_override: Override detected title
        author_override: Override detected author
        hierarchical: If True, create parent (chapter) and child (semantic) chunks.
                      If False, use flat semantic chunking (legacy behavior).
        threshold: Similarity threshold for chunking (0.0-1.0). Lower = fewer breaks.
        min_chunk_size: Minimum words per chunk.
        max_chunk_size: Maximum words per chunk.
        force_reingest: If True, delete existing chunks for this book before ingesting.
        model_id: Embedding model identifier (default: DEFAULT_EMBEDDING_MODEL).

    Returns:
        Dict with success status, chunk counts, and metadata
    """
    # NOTE: Cannot use sys.stderr in Streamlit - it causes [Errno 22]
    logging.debug(f"ingest_book started: {filepath} (collection={collection_name}, hierarchical={hierarchical})")

    try:
        normalized_path, display_path, _, _ = normalize_file_path(filepath)
        logging.debug(f"normalize_file_path returned: normalized_path={repr(normalized_path)}")
    except Exception as e:
        logging.error(f"normalize_file_path FAILED: {e.__class__.__name__}: {e}")
        return {'success': False, 'error': f"Path normalization failed: {e}"}

    ok, err = validate_file_access(normalized_path, display_path)
    if not ok:
        logging.error(f"File access validation failed: {err}")
        return {'success': False, 'error': err}

    # 1. Extract text and metadata
    text, metadata = extract_text(normalized_path)

    logging.debug(f"Text extracted. Title: '{metadata.get('title')}', Author: '{metadata.get('author')}'")
    logging.debug(f"Overrides: title_override={title_override}, author_override={author_override}")

    # Enrich metadata from Calibre ONLY if we don't have overrides
    if not (title_override and author_override):
        logging.debug(f"Running Calibre enrichment (no overrides present)")
        metadata = _enrich_metadata_from_calibre(filepath, metadata)
    else:
        logging.debug(f"Skipping Calibre enrichment (overrides present)")

    # Apply overrides AFTER enrichment
    debug_info = {
        'extracted_author': metadata.get('author'),
        'author_override': author_override,
    }

    if language_override: metadata['language'] = language_override
    if title_override: metadata['title'] = title_override
    if author_override:
        logging.debug(f"Applying author_override: {author_override}")
        debug_info['final_author'] = author_override
        metadata['author'] = author_override
    else:
        debug_info['final_author'] = metadata.get('author')

    # Merge source metadata if provided (e.g., source='gutenberg', source_id=7204)
    if source_meta:
        metadata['source'] = source_meta.get('source', 'unknown')
        metadata['source_id'] = source_meta.get('source_id', '')

    # Log START of ingestion with title and author
    title = metadata.get('title', 'Unknown')
    author = metadata.get('author', 'Unknown')
    logging.info(f"Ingesting: \"{title}\" by {author}")

    # Resolve model_id early for consistent usage
    effective_model_id = model_id or DEFAULT_EMBEDDING_MODEL

    # Force reingest: delete existing chunks first
    if force_reingest:
        deleted_count = delete_book_chunks(
            book_title=title,
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
        if deleted_count > 0:
            logging.info(f"Force reingest: deleted {deleted_count} existing chunks for '{title}'")

    # Setup chunker
    embedder = EmbeddingGenerator()
    if use_semantic:
        chunker = UniversalChunker(
            embedder,
            threshold=threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        logging.info(f"Using SEMANTIC chunking (slow)")
    else:
        chunker = FixedChunker(
            chunk_size=500,
            overlap=50,
            min_chunk_size=100
        )
        logging.info(f"Using FIXED chunking (fast)")

    # Calculate rough sentence count for stats
    sentence_count = len(text.split('. '))

    # Timing
    t_start = time.time()
    t_chunk_start = time.time()

    if hierarchical:
        # ========================================
        # HIERARCHICAL CHUNKING (parent + child)
        # ========================================
        logging.info(f"Using hierarchical chunking for '{metadata.get('title')}'")

        # 2a. Detect chapters
        chapters = detect_chapters(normalized_path, text, metadata)
        logging.info(f"Detected {len(chapters)} chapters via {chapters[0].get('detection_method', 'unknown') if chapters else 'none'}")

        if not chapters:
            # Fallback to single parent
            chapters = [{"title": "Full Book", "text": text, "index": 0, "detection_method": "single"}]

        # 2b. Create parent chunks
        parent_chunks = []
        all_child_chunks = []

        for chapter in chapters:
            parent_id = str(uuid.uuid4())
            chapter_text = chapter.get('text', '')

            # Truncate for embedding (max 8192 tokens)
            embed_text = truncate_for_embedding(chapter_text, max_tokens=8192)

            parent_chunk = {
                'id': parent_id,
                'text': embed_text,
                'full_text': chapter_text,
                'book_title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'language': metadata.get('language', 'unknown'),
                'source': metadata.get('source', 'unknown'),
                'source_id': str(metadata.get('source_id', '')),
                'section_name': chapter.get('title', f"Section {chapter.get('index', 0) + 1}"),
                'section_index': chapter.get('index', 0),
                'token_count': int(len(chapter_text.split()) * 1.3),
                'child_count': 0  # Will be updated after chunking
            }

            # 2c. Create child chunks for this chapter
            chapter_metadata = {
                'parent_id': parent_id,
                'section_name': parent_chunk['section_name'],
                'book_title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'language': metadata.get('language', 'unknown'),
                'source': metadata.get('source', 'unknown'),
                'source_id': str(metadata.get('source_id', '')),
            }

            children = chunker.chunk(chapter_text, metadata=chapter_metadata)

            # Add sequence info to children
            for i, child in enumerate(children):
                child['chunk_level'] = 'child'
                child['parent_id'] = parent_id
                child['sequence_index'] = i
                child['sibling_count'] = len(children)
                child['token_count'] = int(len(child.get('text', '').split()) * 1.3)

            parent_chunk['child_count'] = len(children)
            parent_chunks.append(parent_chunk)
            all_child_chunks.extend(children)

        if not all_child_chunks:
            logging.error(f"No child chunks created for {metadata.get('title')}")
            return {'success': False, 'error': 'No chunks created'}

        logging.info(f"Created {len(parent_chunks)} parent chunks, {len(all_child_chunks)} child chunks")
        t_chunk_end = time.time()

        # 3. Generate embeddings
        t_embed_start = time.time()
        parent_texts = [p['text'] for p in parent_chunks]
        child_texts = [c['text'] for c in all_child_chunks]

        logging.debug(f"Generating embeddings for {len(parent_texts)} parents and {len(child_texts)} children")
        parent_embeddings = generate_embeddings(parent_texts, model_id=effective_model_id)
        child_embeddings = generate_embeddings(child_texts, model_id=effective_model_id)
        t_embed_end = time.time()

        # Filter out failed embeddings (None values from Ollama errors)
        # Convert to list first to handle both numpy arrays and Python lists
        if not isinstance(parent_embeddings, list):
            parent_embeddings = list(parent_embeddings) if parent_embeddings is not None else []
        if not isinstance(child_embeddings, list):
            child_embeddings = list(child_embeddings) if child_embeddings is not None else []
        
        # Filter None values
        if parent_embeddings and any(e is None for e in parent_embeddings):
            filtered_parents = [(c, e) for c, e in zip(parent_chunks, parent_embeddings) if e is not None]
            if filtered_parents:
                parent_chunks, parent_embeddings = zip(*filtered_parents)
                parent_chunks, parent_embeddings = list(parent_chunks), list(parent_embeddings)
            else:
                parent_chunks, parent_embeddings = [], []
        if child_embeddings and any(e is None for e in child_embeddings):
            filtered_children = [(c, e) for c, e in zip(all_child_chunks, child_embeddings) if e is not None]
            if filtered_children:
                all_child_chunks, child_embeddings = zip(*filtered_children)
                all_child_chunks, child_embeddings = list(all_child_chunks), list(child_embeddings)
            else:
                all_child_chunks, child_embeddings = [], []

        # 4. Upload hierarchically (with model metadata)
        t_upload_start = time.time()
        upload_result = upload_hierarchical_to_qdrant(
            parent_chunks, parent_embeddings,
            all_child_chunks, child_embeddings,
            collection_name, qdrant_host, qdrant_port,
            model_id=effective_model_id
        )

        t_upload_end = time.time()

        if not upload_result.get('success'):
            logging.error(f"Hierarchical upload failed: {upload_result.get('error')}")
            return {
                'success': False,
                'error': upload_result.get('error'),
                'title': metadata.get('title', 'Unknown'),
                'filepath': display_path
            }

        result = {
            'success': True,
            'title': metadata.get('title', 'Unknown'),
            'author': metadata.get('author', 'Unknown'),
            'language': metadata.get('language', 'unknown'),
            'source': metadata.get('source', 'unknown'),
            'source_id': str(metadata.get('source_id', '')),
            'chunks': len(all_child_chunks),
            'parent_chunks': len(parent_chunks),
            'child_chunks': len(all_child_chunks),
            'sentences': sentence_count,
            'strategy': 'Hierarchical',
            'chapter_detection': chapters[0].get('detection_method', 'unknown') if chapters else 'none',
            'file_size_mb': os.path.getsize(normalized_path) / (1024 * 1024),
            'filepath': display_path,
            'debug_author': debug_info,
            'hierarchical': True,
            'model_id': effective_model_id
        }

    else:
        # ========================================
        # FLAT CHUNKING (legacy behavior)
        # ========================================
        logging.info(f"Using flat semantic chunking for '{metadata.get('title')}'")

        chunks = chunker.chunk(text, metadata=metadata)

        if not chunks:
            logging.error(f"No chunks created for {metadata.get('title')}")
            return {'success': False, 'error': 'No chunks created'}

        logging.debug(f"Chunks created: {len(chunks)} chunks from {len(text)} characters")
        t_chunk_end = time.time()

        # Embed & Upload (legacy, with model metadata)
        t_embed_start = time.time()
        embeddings = generate_embeddings([c['text'] for c in chunks], model_id=effective_model_id)
        t_embed_end = time.time()

        # Filter out failed embeddings (None values from Ollama errors)
        if not isinstance(embeddings, list):
            embeddings = list(embeddings) if embeddings is not None else []
        if embeddings and any(e is None for e in embeddings):
            filtered = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
            if filtered:
                chunks, embeddings = zip(*filtered)
                chunks, embeddings = list(chunks), list(embeddings)
            else:
                chunks, embeddings = [], []

        t_upload_start = time.time()
        upload_result = upload_to_qdrant(
            chunks, embeddings, collection_name, qdrant_host, qdrant_port,
            model_id=effective_model_id
        )
        t_upload_end = time.time()

        if not upload_result.get('success'):
            logging.error(f"Upload to Qdrant failed: {upload_result.get('error')}")
            return {
                'success': False,
                'error': upload_result.get('error'),
                'title': metadata.get('title', 'Unknown'),
                'filepath': display_path
            }

        result = {
            'success': True,
            'title': metadata.get('title', 'Unknown'),
            'author': metadata.get('author', 'Unknown'),
            'language': metadata.get('language', 'unknown'),
            'source': metadata.get('source', 'unknown'),
            'source_id': str(metadata.get('source_id', '')),
            'chunks': len(chunks),
            'sentences': sentence_count,
            'strategy': 'Universal Semantic',
            'file_size_mb': os.path.getsize(normalized_path) / (1024 * 1024),
            'filepath': display_path,
            'debug_author': debug_info,
            'hierarchical': False,
            'model_id': effective_model_id
        }

    logging.info(f"[OK] Successfully ingested '{result['title']}' ({result['file_size_mb']:.2f} MB, {result['chunks']} chunks)")

    # Update collection manifest
    try:
        manifest = CollectionManifest(collection_name=collection_name)
        manifest.add_book(
            collection_name=collection_name,
            book_path=display_path,
            book_title=result['title'],
            author=result['author'],
            chunks_count=result['chunks'],
            file_size_mb=result['file_size_mb'],
            language=result.get('language'),
            source=result.get('source'),
            source_id=result.get('source_id'),
        )
    except Exception as e:
        logging.warning(f"Failed to update manifest (non-critical): {e}")

    # Log performance to SQLite
    t_end = time.time()
    _device = EmbeddingGenerator().get_model(effective_model_id).device.type
    _batch = 64 if _device == 'cuda' else 32
    _log_ingest_performance(
        book_title=result.get('title', ''),
        author=result.get('author', ''),
        language=result.get('language', ''),
        chunks=result.get('chunks', 0),
        file_size_mb=result.get('file_size_mb', 0),
        duration_total=t_end - t_start,
        duration_embed=t_embed_end - t_embed_start,
        duration_chunk=t_chunk_end - t_chunk_start,
        duration_upload=t_upload_end - t_upload_start,
        device=_device,
        model_id=effective_model_id,
        batch_size=_batch,
        collection=collection_name,
        success=result.get('success', False),
        source=result.get('source', ''),
        source_id=result.get('source_id', ''),
        chunking_mode='semantic' if use_semantic else 'fixed',
    )

    return result

# ============================================================================
# CHUNKING TEST MODE
# ============================================================================

def test_chunking(
    filepath: str,
    threshold: float = 0.55,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1200,
    show_samples: int = 3
) -> Dict:
    """
    Test chunking parameters on a single book WITHOUT uploading to Qdrant.

    Args:
        filepath: Path to the book file
        threshold: Similarity threshold (0.0-1.0). Lower = fewer breaks, Higher = more breaks
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk
        show_samples: Number of sample chunks to include in output

    Returns:
        Dict with statistics and sample chunks
    """
    normalized_path, display_path, _, _ = normalize_file_path(filepath)

    ok, err = validate_file_access(normalized_path, display_path)
    if not ok:
        return {'success': False, 'error': err}

    # Extract text and metadata
    text, metadata = extract_text(normalized_path)
    metadata = _enrich_metadata_from_calibre(filepath, metadata)

    # Apply chunking with specified parameters
    embedder = EmbeddingGenerator()
    chunker = UniversalChunker(
        embedder,
        threshold=threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )

    chunks = chunker.chunk(text, metadata=metadata)

    if not chunks:
        return {'success': False, 'error': 'No chunks created'}

    # Calculate statistics
    word_counts = [c['word_count'] for c in chunks]
    char_counts = [len(c['text']) for c in chunks]

    result = {
        'success': True,
        'file': display_path,
        'title': metadata.get('title', 'Unknown'),
        'author': metadata.get('author', 'Unknown'),
        'parameters': {
            'threshold': threshold,
            'min_chunk_size': min_chunk_size,
            'max_chunk_size': max_chunk_size
        },
        'stats': {
            'total_chunks': len(chunks),
            'total_words': sum(word_counts),
            'total_chars': len(text),
            'avg_words_per_chunk': round(sum(word_counts) / len(chunks), 1),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'avg_chars_per_chunk': round(sum(char_counts) / len(chunks), 1)
        },
        'samples': []
    }

    # Add sample chunks (first N, middle, last)
    sample_indices = []
    if show_samples > 0:
        # First chunk
        sample_indices.append(0)
        # Middle chunk(s)
        if len(chunks) > 2 and show_samples > 1:
            mid = len(chunks) // 2
            sample_indices.append(mid)
        # Last chunk
        if len(chunks) > 1 and show_samples > 2:
            sample_indices.append(len(chunks) - 1)

    for idx in sample_indices[:show_samples]:
        chunk = chunks[idx]
        result['samples'].append({
            'index': idx,
            'word_count': chunk['word_count'],
            'preview': chunk['text'][:500] + ('...' if len(chunk['text']) > 500 else '')
        })

    return result


# ============================================================================
# CHUNKING COMPARISON MODE
# ============================================================================

def compare_chunking(
    filepath: str,
    thresholds: List[float] = None,
    min_chunk_size: int = 200,
    max_chunk_size: int = 1200
) -> Dict:
    """
    Compare multiple threshold values to find optimal semantic chunking parameters.

    Extracts text and generates embeddings ONCE, then tests multiple thresholds
    efficiently. Shows semantic coherence metrics to help choose the best threshold.

    Args:
        filepath: Path to the book file
        thresholds: List of thresholds to test (default: [0.40, 0.45, 0.50, 0.55, 0.60])
        min_chunk_size: Minimum words per chunk
        max_chunk_size: Maximum words per chunk

    Returns:
        Dict with comparison results and recommendation
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import re

    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]

    normalized_path, display_path, _, _ = normalize_file_path(filepath)

    ok, err = validate_file_access(normalized_path, display_path)
    if not ok:
        return {'success': False, 'error': err}

    # Extract text and metadata ONCE
    text, metadata = extract_text(normalized_path)
    metadata = _enrich_metadata_from_calibre(filepath, metadata)

    # Split into sentences ONCE
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

    if len(sentences) < 10:
        return {'success': False, 'error': f'Not enough sentences ({len(sentences)}) for comparison'}

    # Generate embeddings ONCE (expensive operation)
    embedder = EmbeddingGenerator()
    if hasattr(embedder, 'generate_embeddings'):
        embeddings = np.array(embedder.generate_embeddings(sentences))
    else:
        embeddings = embedder.model.encode(sentences, show_progress_bar=False)

    # Calculate pairwise similarities between adjacent sentences
    adjacent_similarities = []
    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            embeddings[i-1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]
        adjacent_similarities.append(sim)

    # Test each threshold
    results = []
    for threshold in sorted(thresholds):
        chunks = []
        current_sentences = [sentences[0]]
        current_word_count = len(sentences[0].split())
        break_points = []  # (index, similarity, before_text, after_text)

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            word_count = len(sentence.split())
            similarity = adjacent_similarities[i-1]

            should_break = (similarity < threshold and current_word_count >= min_chunk_size)
            must_break = (current_word_count + word_count > max_chunk_size)

            if should_break or must_break:
                # Record break point
                before_text = sentences[i-1][-80:] if len(sentences[i-1]) > 80 else sentences[i-1]
                after_text = sentence[:80] if len(sentence) > 80 else sentence
                break_points.append({
                    'index': i,
                    'similarity': round(similarity, 3),
                    'reason': 'semantic' if should_break else 'max_size',
                    'before': '...' + before_text if len(sentences[i-1]) > 80 else before_text,
                    'after': after_text + '...' if len(sentence) > 80 else after_text
                })

                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_word_count = word_count
            else:
                current_sentences.append(sentence)
                current_word_count += word_count

        # Final chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # Calculate chunk statistics
        word_counts = [len(c.split()) for c in chunks]

        # Calculate intra-chunk coherence (avg similarity within chunks)
        chunk_coherences = []
        sent_idx = 0
        for chunk in chunks:
            chunk_sents = re.split(r'(?<=[.!?])\s+', chunk)
            chunk_sents = [s.strip() for s in chunk_sents if len(s.strip()) > 2]
            n_sents = len(chunk_sents)

            if n_sents > 1:
                # Average similarity of adjacent sentences within this chunk
                chunk_sims = adjacent_similarities[sent_idx:sent_idx + n_sents - 1]
                if chunk_sims:
                    chunk_coherences.append(np.mean(chunk_sims))
            sent_idx += n_sents

        avg_coherence = round(np.mean(chunk_coherences), 3) if chunk_coherences else 0

        # Count semantic vs forced breaks
        semantic_breaks = sum(1 for bp in break_points if bp['reason'] == 'semantic')
        forced_breaks = sum(1 for bp in break_points if bp['reason'] == 'max_size')

        results.append({
            'threshold': threshold,
            'chunks': len(chunks),
            'avg_words': round(np.mean(word_counts), 1),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'coherence': avg_coherence,
            'semantic_breaks': semantic_breaks,
            'forced_breaks': forced_breaks,
            'sample_breaks': break_points[:3]  # First 3 break points as samples
        })

    # Recommendation logic: PURE SEMANTIC QUALITY
    # Forced breaks destroy content integrity - they cut mid-topic to meet size limits
    # Coherence measures how semantically related sentences are within chunks
    # NO preference for chunk count - that's a symptom, not a goal
    best = None
    best_score = -1
    for r in results:
        # Coherence is king, forced breaks are the enemy of content integrity
        score = r['coherence'] * 100 - (r['forced_breaks'] * 20)
        if score > best_score:
            best_score = score
            best = r['threshold']

    # Build recommendation reason
    best_result = next(r for r in results if r['threshold'] == best)
    if best_result['forced_breaks'] == 0:
        reason = f"Highest coherence ({best_result['coherence']}) with pure semantic breaks"
    else:
        reason = f"Best coherence ({best_result['coherence']}) but {best_result['forced_breaks']} forced breaks - consider increasing max_chunk_size"

    return {
        'success': True,
        'file': display_path,
        'title': metadata.get('title', 'Unknown'),
        'author': metadata.get('author', 'Unknown'),
        'total_sentences': len(sentences),
        'total_words': sum(len(s.split()) for s in sentences),
        'similarity_distribution': {
            'min': round(min(adjacent_similarities), 3),
            'max': round(max(adjacent_similarities), 3),
            'mean': round(np.mean(adjacent_similarities), 3),
            'median': round(np.median(adjacent_similarities), 3)
        },
        'comparisons': results,
        'recommendation': best,
        'recommendation_reason': reason
    }


def main():
    parser = argparse.ArgumentParser(description='Alexandria Book Ingestion')
    parser.add_argument('--file', required=True, help='Path to book file')
    parser.add_argument('--collection', default='alexandria', help='Qdrant collection name')
    parser.add_argument('--host', default=QDRANT_HOST, help=f'Qdrant host (default: {QDRANT_HOST})')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')

    # Chunking test mode
    parser.add_argument('--dry-run', action='store_true',
                        help='Test chunking without uploading to Qdrant')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple thresholds to find optimal semantic chunking')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Similarity threshold (0.0-1.0). Lower=fewer breaks')
    parser.add_argument('--min-chunk', type=int, default=200,
                        help='Minimum words per chunk')
    parser.add_argument('--max-chunk', type=int, default=1200,
                        help='Maximum words per chunk')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of sample chunks to show')

    args = parser.parse_args()

    if args.compare:
        # Compare mode - test multiple thresholds
        result = compare_chunking(
            args.file,
            min_chunk_size=args.min_chunk,
            max_chunk_size=args.max_chunk
        )

        if not result['success']:
            print(f"[ERROR] {result['error']}")
            return

        print("\n" + "=" * 78)
        print(f"SEMANTIC CHUNKING COMPARISON: {result['title']}")
        print(f"Author: {result['author']}")
        print("=" * 78)

        print(f"\nText Statistics:")
        print(f"  Sentences: {result['total_sentences']:,}")
        print(f"  Words:     {result['total_words']:,}")

        sim = result['similarity_distribution']
        print(f"\nAdjacent Sentence Similarity Distribution:")
        print(f"  Min: {sim['min']:.3f}  Max: {sim['max']:.3f}  Mean: {sim['mean']:.3f}  Median: {sim['median']:.3f}")

        print(f"\n{'Threshold':>10} {'Chunks':>8} {'Avg Words':>10} {'Coherence':>10} {'Semantic':>10} {'Forced':>8}")
        print("-" * 68)
        for r in result['comparisons']:
            marker = " <--" if r['threshold'] == result['recommendation'] else ""
            print(f"{r['threshold']:>10.2f} {r['chunks']:>8} {r['avg_words']:>10.1f} {r['coherence']:>10.3f} {r['semantic_breaks']:>10} {r['forced_breaks']:>8}{marker}")

        print(f"\nRECOMMENDATION: threshold={result['recommendation']}")
        print(f"  {result['recommendation_reason']}")

        # Show sample break points from recommended threshold
        rec_data = next(r for r in result['comparisons'] if r['threshold'] == result['recommendation'])
        if rec_data['sample_breaks']:
            print(f"\nSample Break Points (threshold={result['recommendation']}):")
            for i, bp in enumerate(rec_data['sample_breaks'][:2], 1):
                print(f"\n  Break #{i} (similarity={bp['similarity']}, reason={bp['reason']}):")
                print(f"    BEFORE: \"{bp['before']}\"")
                print(f"    AFTER:  \"{bp['after']}\"")

        print("\n" + "=" * 78)
        print("[OK] Comparison complete. Use --threshold X for ingestion.")
        print("=" * 78 + "\n")

    elif args.dry_run:
        # Test mode - no upload
        result = test_chunking(
            args.file,
            threshold=args.threshold,
            min_chunk_size=args.min_chunk,
            max_chunk_size=args.max_chunk,
            show_samples=args.samples
        )

        if not result['success']:
            print(f"[ERROR] Error: {result['error']}")
            return

        print("\n" + "=" * 70)
        print(f"CHUNKING TEST: {result['title']}")
        print(f"   Author: {result['author']}")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"   threshold:      {result['parameters']['threshold']}")
        print(f"   min_chunk_size: {result['parameters']['min_chunk_size']} words")
        print(f"   max_chunk_size: {result['parameters']['max_chunk_size']} words")
        print(f"\nStatistics:")
        stats = result['stats']
        print(f"   Total chunks:   {stats['total_chunks']}")
        print(f"   Total words:    {stats['total_words']:,}")
        print(f"   Avg words/chunk: {stats['avg_words_per_chunk']}")
        print(f"   Min words:      {stats['min_words']}")
        print(f"   Max words:      {stats['max_words']}")

        if result['samples']:
            print(f"\nSample Chunks:")
            for sample in result['samples']:
                print(f"\n--- Chunk #{sample['index']} ({sample['word_count']} words) ---")
                print(sample['preview'])

        print("\n" + "=" * 70)
        print("[OK] Dry run complete. No data uploaded to Qdrant.")
        print("=" * 70 + "\n")
    else:
        # Normal ingest mode
        ingest_book(args.file, args.collection, args.host, args.port)

if __name__ == '__main__':
    main()