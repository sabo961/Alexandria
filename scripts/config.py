#!/usr/bin/env python3
"""
Alexandria Configuration
========================

Central configuration loader for all Alexandria scripts.
All values come from .env file (or environment variables).

Priority:
1. Environment variables (highest)
2. .env file in project root
3. Generic fallback defaults (lowest)

Setup:
    cp .env.example .env
    # Edit .env with your values
"""

import os
import sys
from pathlib import Path

# Find project root (where .env should be)
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# Load .env file if it exists
ENV_FILE = PROJECT_ROOT / '.env'
if ENV_FILE.exists():
    with open(ENV_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value

# =============================================================================
# QDRANT CONFIGURATION
# =============================================================================

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))
QDRANT_COLLECTION = os.environ.get('QDRANT_COLLECTION', 'alexandria')

# =============================================================================
# CALIBRE CONFIGURATION
# =============================================================================

CALIBRE_LIBRARY_PATH = os.environ.get('CALIBRE_LIBRARY_PATH', '')
CALIBRE_WEB_URL = os.environ.get('CALIBRE_WEB_URL', '')
CWA_INGEST_PATH = os.environ.get('CWA_INGEST_PATH', '')

# =============================================================================
# LOCAL FILE INGESTION
# =============================================================================

LOCAL_INGEST_PATH = os.environ.get('LOCAL_INGEST_PATH', str(Path.home() / 'Downloads'))

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================

EMBEDDING_MODELS = {
    "minilm": {"name": "all-MiniLM-L6-v2", "dim": 384},
    "bge-large": {"name": "BAAI/bge-large-en-v1.5", "dim": 1024},
    "bge-m3": {"name": "BAAI/bge-m3", "dim": 1024},  # Multilingual (100+ languages)
}
DEFAULT_EMBEDDING_MODEL = os.environ.get('DEFAULT_EMBEDDING_MODEL', 'bge-m3')
EMBEDDING_DEVICE = os.environ.get('EMBEDDING_DEVICE', 'auto')  # auto, cuda, cpu

# =============================================================================
# ALEXANDRIA DATABASE (shared SQLite for ingest log + manifest)
# =============================================================================

# Default: .qdrant/ subfolder inside Calibre library (shared across machines)
# Fallback: local logs/ folder if CALIBRE_LIBRARY_PATH not set
_default_db = ''
if CALIBRE_LIBRARY_PATH:
    _default_db = str(Path(CALIBRE_LIBRARY_PATH) / '.qdrant' / 'alexandria.db')
ALEXANDRIA_DB = os.environ.get('ALEXANDRIA_DB', _default_db)

# =============================================================================
# GUARDIAN PERSONAS
# =============================================================================

# Directory containing guardian .md files with alexandria: frontmatter
# Default: docs/development/guardians/ relative to project root
GUARDIANS_DIR = os.environ.get(
    'GUARDIANS_DIR',
    str(PROJECT_ROOT / 'docs' / 'development' / 'guardians')
)

# =============================================================================
# INGESTION VERSIONING
# =============================================================================

INGEST_VERSION = "2.0"  # Semantic version for tracking ingestion schema changes

# =============================================================================
# PHOENIX OBSERVABILITY
# =============================================================================

PHOENIX_ENABLED = os.environ.get('PHOENIX_ENABLED', 'true').lower() != 'false'
PHOENIX_PORT = int(os.environ.get('PHOENIX_PORT', '6006'))

# =============================================================================
# OPENROUTER (OPTIONAL - for CLI testing)
# =============================================================================

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

# Try to load from .streamlit/secrets.toml if not in env
if not OPENROUTER_API_KEY:
    secrets_file = PROJECT_ROOT / '.streamlit' / 'secrets.toml'
    if secrets_file.exists():
        try:
            import tomllib
            with open(secrets_file, 'rb') as f:
                secrets = tomllib.load(f)
            OPENROUTER_API_KEY = secrets.get('OPENROUTER_API_KEY', '')
        except Exception:
            pass

# =============================================================================
# VALIDATION
# =============================================================================

_MISSING = []
if not QDRANT_HOST or QDRANT_HOST == 'localhost':
    pass  # localhost is a valid default for local dev
if not CALIBRE_LIBRARY_PATH:
    _MISSING.append('CALIBRE_LIBRARY_PATH')
if not CWA_INGEST_PATH:
    _MISSING.append('CWA_INGEST_PATH')

if _MISSING and not any(arg in sys.argv for arg in ['--help', '-h']):
    if not ENV_FILE.exists():
        print(f"[WARN] No .env file found at: {ENV_FILE}")
        print(f"       Run: cp .env.example .env  (then edit with your values)")
    else:
        print(f"[WARN] Missing config in .env: {', '.join(_MISSING)}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_qdrant_url() -> str:
    """Get full Qdrant URL."""
    return f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def print_config():
    """Print current configuration (for debugging)."""
    print("Alexandria Configuration")
    print("=" * 40)
    print(f"ENV_FILE:             {ENV_FILE} ({'exists' if ENV_FILE.exists() else 'NOT FOUND'})")
    print(f"QDRANT_HOST:          {QDRANT_HOST}")
    print(f"QDRANT_PORT:          {QDRANT_PORT}")
    print(f"QDRANT_COLLECTION:    {QDRANT_COLLECTION}")
    print(f"CALIBRE_LIBRARY_PATH: {CALIBRE_LIBRARY_PATH or '(not set)'}")
    print(f"CALIBRE_WEB_URL:      {CALIBRE_WEB_URL or '(not set)'}")
    print(f"CWA_INGEST_PATH:      {CWA_INGEST_PATH or '(not set)'}")
    print(f"LOCAL_INGEST_PATH:    {LOCAL_INGEST_PATH}")
    print(f"EMBEDDING_MODELS:     {list(EMBEDDING_MODELS.keys())}")
    print(f"DEFAULT_MODEL:        {DEFAULT_EMBEDDING_MODEL}")
    print(f"EMBEDDING_DEVICE:     {EMBEDDING_DEVICE}")
    print(f"ALEXANDRIA_DB:        {ALEXANDRIA_DB or '(not set - using local fallback)'}")
    print(f"INGEST_VERSION:       {INGEST_VERSION}")
    print(f"OPENROUTER_API_KEY:   {'***' + OPENROUTER_API_KEY[-4:] if OPENROUTER_API_KEY else '(not set)'}")
    print(f"PHOENIX_ENABLED:      {PHOENIX_ENABLED}")
    print(f"PHOENIX_PORT:         {PHOENIX_PORT}")
    print("=" * 40)


if __name__ == "__main__":
    print_config()
