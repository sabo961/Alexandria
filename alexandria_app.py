#!/usr/bin/env python3
"""
Alexandria of Temenos - Dashboard
Simplified single-page interface for library management and RAG queries.

Launch:
    streamlit run alexandria_app.py
"""

import streamlit as st

# Hide Deploy button (keep Settings)
st.markdown("""
<style>
    [data-testid="stAppDeployButton"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)
import sys
import json
import requests
from pathlib import Path

# Add scripts to path
project_root = Path(__file__).parent
scripts_root = project_root / "scripts"
if str(scripts_root) not in sys.path:
    sys.path.insert(0, str(scripts_root))

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    CALIBRE_LIBRARY_PATH, OPENROUTER_API_KEY, ALEXANDRIA_DB
)
from qdrant_utils import check_qdrant_connection, list_collections
from calibre_db import CalibreDB
from rag_query import perform_rag_query
from collection_manifest import CollectionManifest

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Alexandria",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=60)
def check_qdrant_status():
    """Check Qdrant connection (cached for 60s)."""
    connected, error = check_qdrant_connection(QDRANT_HOST, QDRANT_PORT)
    return connected, error

@st.cache_data(ttl=300)
def get_collection_stats():
    """Get collection statistics (cached for 5min)."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        stats = {}
        for coll in collections:
            info = client.get_collection(coll.name)
            stats[coll.name] = info.points_count
        return stats
    except Exception as e:
        return {"error": str(e)}

def _calibre_db_mtime() -> float:
    """Return metadata.db mtime so cache auto-invalidates on Calibre changes."""
    try:
        return Path(CALIBRE_LIBRARY_PATH, "metadata.db").stat().st_mtime
    except OSError:
        return 0.0

@st.cache_data(ttl=300)
def load_calibre_books(_mtime: float = 0.0):
    """Load books from Calibre (cached until metadata.db changes or 5min TTL)."""
    try:
        db = CalibreDB(CALIBRE_LIBRARY_PATH)
        return db.get_all_books()
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=60)
def load_manifest(collection_name: str):
    """Load manifest for collection from SQLite."""
    try:
        manifest = CollectionManifest(collection_name=collection_name)
        books = manifest.get_books(collection_name)
        if not books:
            return None
        summary = manifest.get_summary(collection_name)
        return {
            'books': books,
            'total_chunks': summary.get('total_chunks', 0),
            'total_size_mb': summary.get('total_size_mb', 0),
        }
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_books_from_qdrant(collection_name: str):
    """Fallback: Get book list directly from Qdrant payloads."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Scroll through collection to get unique books
        books = {}
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                p = point.payload
                book_key = (p.get('book_title', 'Unknown'), p.get('author', 'Unknown'))
                if book_key not in books:
                    books[book_key] = {
                        'book_title': p.get('book_title', 'Unknown'),
                        'author': p.get('author', 'Unknown'),
                        'language': p.get('language', '?'),
                        'chunks_count': 0
                    }
                books[book_key]['chunks_count'] += 1

            if offset is None:
                break

        return list(books.values())
    except Exception as e:
        return None

def load_prompt_patterns():
    """Load prompt patterns from JSON."""
    patterns_file = project_root / "prompts" / "patterns.json"
    if patterns_file.exists():
        with open(patterns_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# =============================================================================
# SIDEBAR - Configuration & Status
# =============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")

    # Connection Status
    st.subheader("Connection Status")
    qdrant_ok, qdrant_error = check_qdrant_status()

    if qdrant_ok:
        st.success(f"🟢 Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    else:
        st.error(f"🔴 Qdrant: Disconnected")
        with st.expander("Error details"):
            st.code(qdrant_error)

    # Calibre status
    calibre_path = Path(CALIBRE_LIBRARY_PATH)
    if calibre_path.exists():
        st.success(f"🟢 Calibre: Connected")
        st.caption(f"📁 {CALIBRE_LIBRARY_PATH}")
    else:
        st.error(f"🔴 Calibre: Path not found")

    st.divider()

    # Quick Stats - only show manifest-tracked collections
    st.subheader("📊 Quick Stats")

    if qdrant_ok:
        stats = get_collection_stats()
        try:
            _manifest = CollectionManifest()
            _book_collections = set(_manifest.list_collection_names())
            # Get ingested book count from manifest
            _ingested_books = _manifest.count_books(QDRANT_COLLECTION)
        except Exception:
            _book_collections = {QDRANT_COLLECTION}
            _ingested_books = 0

        if "error" not in stats:
            for coll_name, count in stats.items():
                if coll_name in _book_collections:
                    st.metric(f"📦 {coll_name}", f"{count:,} chunks")
            # Show ingested books count
            if _ingested_books > 0:
                st.metric("✅ Ingested Books", f"{_ingested_books:,}")
        else:
            st.warning("Could not load stats")

    # Calibre book count
    books = load_calibre_books(_mtime=_calibre_db_mtime())
    if books and not isinstance(books, tuple):
        st.metric("📚 Calibre Library", f"{len(books):,} books")

    st.divider()

    # OpenRouter Settings (as fragment to avoid full page reruns)
    @st.fragment
    def openrouter_settings():
        st.subheader("🤖 OpenRouter")

        if OPENROUTER_API_KEY:
            # Fetch Models button
            if st.button("🔄 Fetch Models", width="stretch", key="fetch_models"):
                with st.spinner("Fetching models..."):
                    try:
                        response = requests.get(
                            "https://openrouter.ai/api/v1/models",
                            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
                        )
                        if response.status_code == 200:
                            models_data = response.json().get("data", [])
                            openrouter_models = {}
                            for model in models_data:
                                model_id = model.get("id", "")
                                model_name = model.get("name", model_id)
                                pricing = model.get("pricing", {})
                                prompt_price = float(pricing.get("prompt", "1") or "1")
                                is_free = prompt_price == 0
                                emoji = "🆓" if is_free else "💰"
                                display_name = f"{emoji} {model_name}"
                                openrouter_models[display_name] = model_id

                            # Sort: free first, then alphabetically
                            sorted_models = dict(sorted(
                                openrouter_models.items(),
                                key=lambda x: (not x[0].startswith("🆓"), x[0])
                            ))
                            st.session_state['openrouter_models'] = sorted_models
                            st.success(f"✅ {len(sorted_models)} models loaded")
                        else:
                            st.error(f"API error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # Model dropdown (if models fetched)
            if 'openrouter_models' in st.session_state and st.session_state['openrouter_models']:
                models = st.session_state['openrouter_models']

                # Try to restore last selection
                default_idx = 0
                if 'selected_model_name' in st.session_state:
                    try:
                        default_idx = list(models.keys()).index(st.session_state['selected_model_name'])
                    except ValueError:
                        default_idx = 0

                selected_name = st.selectbox(
                    "Model",
                    list(models.keys()),
                    index=default_idx,
                    help="🆓 = Free models",
                    key="model_select"
                )
                st.session_state['selected_model_name'] = selected_name
                st.session_state['selected_model'] = models[selected_name]
            else:
                st.caption("Click 'Fetch Models' to load available models")
                st.session_state['selected_model'] = None
        else:
            st.warning("🔑 No API key")
            st.caption("Add OPENROUTER_API_KEY to .env")
            st.session_state['selected_model'] = None

    openrouter_settings()

    st.divider()

    # Refresh button
    if st.button("🔄 Refresh All", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# MAIN AREA
# =============================================================================
_logo_path = project_root / "assets" / "logo.png"
if _logo_path.exists():
    _left, _right = st.columns([0.07, 0.93])
    with _left:
        st.image(str(_logo_path), width=64)
    with _right:
        st.title("Alexandria of Temenos")
else:
    st.title("📚 Alexandria of Temenos")
st.caption("Knowledge Management Dashboard")

# =============================================================================
# SECTION 1: Calibre Library
# =============================================================================
with st.expander("📚 Calibre Library", expanded=False):
    books = load_calibre_books(_mtime=_calibre_db_mtime())

    if books is None or isinstance(books, tuple):
        st.error(f"Could not connect to Calibre: {books[1] if isinstance(books, tuple) else 'Unknown error'}")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            # Author filter — text input to avoid rendering thousands of items in a dropdown
            author_filter = st.text_input("Filter by author", key="calibre_author", placeholder="e.g. Pessoa")

        with col2:
            # Language filter
            languages = sorted(set(b.language for b in books))
            selected_lang = st.selectbox("Language", ["All"] + languages, key="calibre_lang")

        with col3:
            # Search
            search_term = st.text_input("Search title", key="calibre_search")

        # Filter books
        filtered = books
        if author_filter:
            filtered = [b for b in filtered if author_filter.lower() in b.author.lower()]
        if selected_lang != "All":
            filtered = [b for b in filtered if b.language == selected_lang]
        if search_term:
            filtered = [b for b in filtered if search_term.lower() in b.title.lower()]

        st.caption(f"Showing {len(filtered)} of {len(books)} books")

        # Display as table
        if filtered:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Title": b.title,
                    "Author": b.author,
                    "Language": b.language,
                    "Formats": ", ".join(b.formats),
                    "Tags": ", ".join(b.tags[:3]) if b.tags else ""
                }
                for b in filtered[:100]  # Limit to 100 for performance
            ])
            st.dataframe(df, width="stretch", hide_index=True)

            if len(filtered) > 100:
                st.caption("Showing first 100 results. Use filters to narrow down.")

# =============================================================================
# SECTION 2: Ingested Books
# =============================================================================
with st.expander("📖 Ingested Books (Qdrant)", expanded=False):
    if not qdrant_ok:
        st.error("Qdrant not connected")
    else:
        # Show only collections that have books in the manifest
        try:
            manifest_browser = CollectionManifest()
            manifest_collections = manifest_browser.list_collection_names()
        except Exception:
            manifest_collections = []

        if not manifest_collections:
            manifest_collections = [QDRANT_COLLECTION]

        if len(manifest_collections) == 1:
            selected_coll = manifest_collections[0]
        else:
            default_idx = manifest_collections.index(QDRANT_COLLECTION) if QDRANT_COLLECTION in manifest_collections else 0
            selected_coll = st.selectbox("Collection", manifest_collections, index=default_idx, key="ingested_coll")

        # Load manifest for collection
        manifest_data = load_manifest(selected_coll)

        if manifest_data and manifest_data.get('books'):
            books_data = manifest_data.get('books', [])

            # --- Filters (same pattern as Calibre Library) ---
            fcol1, fcol2, fcol3 = st.columns([2, 1, 2])
            iq_author_filter = fcol1.text_input("Filter by author", key="iq_author", placeholder="e.g. Pessoa")
            # Build language options from data
            _iq_langs = sorted(set(b.get('language', '?') for b in books_data))
            iq_lang_filter = fcol2.selectbox("Language", ["All"] + _iq_langs, key="iq_lang")
            iq_title_filter = fcol3.text_input("Search title", key="iq_title")

            filtered_data = books_data
            if iq_author_filter:
                filtered_data = [b for b in filtered_data if iq_author_filter.lower() in (b.get('author', '') or '').lower()]
            if iq_lang_filter != "All":
                filtered_data = [b for b in filtered_data if b.get('language', '?') == iq_lang_filter]
            if iq_title_filter:
                filtered_data = [b for b in filtered_data if iq_title_filter.lower() in (b.get('book_title', '') or '').lower()]

            total_chunks = sum(b.get('chunks_count', 0) for b in filtered_data)
            st.metric("Total Chunks", f"{total_chunks:,}")
            st.caption(f"📋 Collection: {selected_coll} | Source: SQLite manifest | Showing {len(filtered_data)} of {len(books_data)} books")

            import pandas as pd

            def _fmt_source(b):
                src = b.get('source', '') or ''
                sid = b.get('source_id', '') or ''
                if not src or src == 'unknown':
                    return ''
                return f"{src} #{sid}" if sid else src

            df = pd.DataFrame([
                {
                    "Title": b.get('book_title', 'Unknown'),
                    "Author": b.get('author', 'Unknown'),
                    "Language": b.get('language', '?'),
                    "Chunks": b.get('chunks_count', 0),
                    "Source": _fmt_source(b),
                    "Ingested": b.get('ingested_at', '')[:10]
                }
                for b in filtered_data
            ])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            # Fallback: Query Qdrant directly
            with st.spinner("Scanning collection..."):
                books_data = get_books_from_qdrant(selected_coll)

            if books_data:
                # --- Filters ---
                fcol1, fcol2, fcol3 = st.columns([2, 1, 2])
                iqf_author = fcol1.text_input("Filter by author", key="iqf_author", placeholder="e.g. Pessoa")
                _iqf_langs = sorted(set(b.get('language', '?') for b in books_data))
                iqf_lang = fcol2.selectbox("Language", ["All"] + _iqf_langs, key="iqf_lang")
                iqf_title = fcol3.text_input("Search title", key="iqf_title")

                filtered_data = books_data
                if iqf_author:
                    filtered_data = [b for b in filtered_data if iqf_author.lower() in (b.get('author', '') or '').lower()]
                if iqf_lang != "All":
                    filtered_data = [b for b in filtered_data if b.get('language', '?') == iqf_lang]
                if iqf_title:
                    filtered_data = [b for b in filtered_data if iqf_title.lower() in (b.get('book_title', '') or '').lower()]

                total_chunks = sum(b['chunks_count'] for b in filtered_data)
                st.metric("Total Chunks", f"{total_chunks:,}")
                st.caption(f"📋 Collection: {selected_coll} | Source: Qdrant (no manifest) | Showing {len(filtered_data)} of {len(books_data)} books")

                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Title": b.get('book_title', 'Unknown'),
                        "Author": b.get('author', 'Unknown'),
                        "Chunks": b.get('chunks_count', 0),
                        "Language": b.get('language', '?'),
                    }
                    for b in filtered_data
                ])
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("Collection is empty or could not be read.")

# =============================================================================
# SECTION 3: Ingest Log
# =============================================================================
with st.expander("📊 Ingest Log", expanded=False):
    try:
        import sqlite3
        from pathlib import Path

        db_path = ALEXANDRIA_DB if ALEXANDRIA_DB else str(Path(__file__).parent / 'logs' / 'alexandria.db')
        log_collection = selected_coll if 'selected_coll' in dir() else QDRANT_COLLECTION

        if Path(db_path).exists():
            # Limit selector
            col_limit1, col_limit2 = st.columns([1, 3])
            with col_limit1:
                show_all = st.checkbox("Show all", value=False, key="log_show_all")
            with col_limit2:
                if not show_all:
                    limit_val = st.select_slider("Last N", options=[50, 100, 200, 500], value=100, key="log_limit")
                else:
                    limit_val = None
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            if show_all:
                rows = conn.execute(
                    '''SELECT timestamp, hostname, book_title, author, language,
                              chunks, duration_total, duration_embed, chunks_per_sec,
                              device, collection, success, chunking_mode
                       FROM ingest_log WHERE collection=?
                       ORDER BY timestamp DESC''',
                    (log_collection,)
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT timestamp, hostname, book_title, author, language,
                              chunks, duration_total, duration_embed, chunks_per_sec,
                              device, collection, success, chunking_mode
                       FROM ingest_log WHERE collection=?
                       ORDER BY timestamp DESC LIMIT ?''',
                    (log_collection, limit_val)
                ).fetchall()
            conn.close()

            if rows:
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Date": r['timestamp'][:16].replace('T', ' '),
                        "Host": r['hostname'] or '?',
                        "Book": r['book_title'] or '?',
                        "Author": r['author'] or '?',
                        "Lang": r['language'] or '?',
                        "Mode": r['chunking_mode'] or 'semantic',
                        "Chunks": r['chunks'],
                        "Total (s)": round(r['duration_total'], 1) if r['duration_total'] else 0,
                        "Embed (s)": round(r['duration_embed'], 1) if r['duration_embed'] else 0,
                        "Ch/sec": round(r['chunks_per_sec'], 1) if r['chunks_per_sec'] else 0,
                        "Device": r['device'] or '?',
                        "OK": "yes" if r['success'] else "no",
                    }
                    for r in rows
                ])
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption(f"Collection: {log_collection} | Last {len(rows)} jobs")
            else:
                st.info(f"No ingest jobs for '{log_collection}'.")
        else:
            st.info(f"Database not found: {db_path}")
    except Exception as e:
        st.error(f"Could not load ingest log: {e}")

# =============================================================================
# SECTION: Chunking Rules (DEPRECATED)
# =============================================================================
# Removed - use Author Curation tab instead (SQLite-based)

# =============================================================================
# SECTION 3a: Author Curation
# =============================================================================

with st.expander("👤 Author Curation", expanded=False):
    st.caption("Manage author → chunking mode mapping. Only 'semantic' and 'fixed' authors get ingested.")
    
    try:
        import sqlite3
        
        if not ALEXANDRIA_DB:
            st.warning("ALEXANDRIA_DB not configured.")
        else:
            conn = sqlite3.connect(ALEXANDRIA_DB)
            conn.row_factory = sqlite3.Row
            
            # Stats
            stats = conn.execute('''
                SELECT mode, COUNT(*) as cnt, SUM(book_count) as books
                FROM author_chunking GROUP BY mode
            ''').fetchall()
            
            col1, col2, col3, col4 = st.columns(4)
            for row in stats:
                if row['mode'] == 'semantic':
                    col1.metric("🟢 Semantic", f"{row['cnt']} ({row['books'] or 0} books)")
                elif row['mode'] == 'fixed':
                    col2.metric("🔵 Fixed", f"{row['cnt']} ({row['books'] or 0} books)")
                elif row['mode'] == 'none':
                    col3.metric("⚪ None", f"{row['cnt']} ({row['books'] or 0} books)")
            
            st.divider()
            
            # Filters
            fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
            search = fcol1.text_input("🔍 Search author", key="author_search", placeholder="e.g. Jung")
            mode_filter = fcol2.selectbox("Mode", ["all", "semantic", "fixed", "none"], key="author_mode_filter")
            page_size = fcol3.selectbox("Per page", [25, 50, 100], key="author_page_size")
            
            # Pagination
            if 'author_page' not in st.session_state:
                st.session_state.author_page = 0
            
            # Build query
            query = "SELECT author_sort, mode, book_count FROM author_chunking WHERE 1=1"
            params = []
            if search:
                query += " AND author_sort LIKE ?"
                params.append(f"%{search}%")
            if mode_filter != "all":
                query += " AND mode = ?"
                params.append(mode_filter)
            
            # Count total
            count_query = query.replace("SELECT author_sort, mode, book_count", "SELECT COUNT(*)")
            total = conn.execute(count_query, params).fetchone()[0]
            total_pages = max(1, (total + page_size - 1) // page_size)
            
            # Clamp page
            st.session_state.author_page = max(0, min(st.session_state.author_page, total_pages - 1))
            offset = st.session_state.author_page * page_size
            
            # Fetch page
            query += f" ORDER BY author_sort LIMIT {page_size} OFFSET {offset}"
            authors = conn.execute(query, params).fetchall()
            
            # Pagination controls
            pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
            if pcol1.button("⬅️ Prev", disabled=st.session_state.author_page == 0):
                st.session_state.author_page -= 1
                st.rerun()
            pcol2.write(f"Page {st.session_state.author_page + 1} of {total_pages} ({total} authors)")
            if pcol3.button("Next ➡️", disabled=st.session_state.author_page >= total_pages - 1):
                st.session_state.author_page += 1
                st.rerun()
            
            # Display authors with toggle buttons
            for row in authors:
                author = row['author_sort']
                current_mode = row['mode']
                books = row['book_count'] or 0
                
                acol1, acol2, acol3, acol4, acol5 = st.columns([4, 1, 1, 1, 1])
                acol1.write(f"**{author}** ({books} books)")
                
                # Mode buttons
                if acol2.button("🟢", key=f"sem_{author}", help="Set semantic", 
                               disabled=current_mode == 'semantic', type="primary" if current_mode == 'semantic' else "secondary"):
                    conn.execute("UPDATE author_chunking SET mode='semantic' WHERE author_sort=?", (author,))
                    conn.commit()
                    st.rerun()
                
                if acol3.button("🔵", key=f"fix_{author}", help="Set fixed",
                               disabled=current_mode == 'fixed', type="primary" if current_mode == 'fixed' else "secondary"):
                    conn.execute("UPDATE author_chunking SET mode='fixed' WHERE author_sort=?", (author,))
                    conn.commit()
                    st.rerun()
                
                if acol4.button("⚪", key=f"non_{author}", help="Set none (skip)",
                               disabled=current_mode == 'none', type="primary" if current_mode == 'none' else "secondary"):
                    conn.execute("UPDATE author_chunking SET mode='none' WHERE author_sort=?", (author,))
                    conn.commit()
                    st.rerun()
                
                acol5.write(f"`{current_mode}`")
            
            # --- Title Contains Section ---
            st.divider()
            st.subheader("📝 Title Contains")
            st.caption("Patterns that match ANY title, regardless of author. Add patterns like 'alchemy', 'kabbalah', etc.")
            
            # Check if title_chunking table exists
            has_title_table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='title_chunking'"
            ).fetchone()
            
            if has_title_table:
                # Add new pattern
                tcol1, tcol2, tcol3 = st.columns([3, 1, 1])
                new_pattern = tcol1.text_input("New pattern", key="new_title_pattern", placeholder="e.g. alchemy")
                new_mode = tcol2.selectbox("Mode", ["semantic", "fixed"], key="new_title_mode")
                if tcol3.button("➕ Add", key="add_title_pattern"):
                    if new_pattern.strip():
                        try:
                            from datetime import datetime
                            conn.execute(
                                "INSERT OR REPLACE INTO title_chunking (pattern, mode, updated_at) VALUES (?, ?, ?)",
                                (new_pattern.strip().lower(), new_mode, datetime.now().isoformat())
                            )
                            conn.commit()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                # List existing patterns
                patterns = conn.execute(
                    "SELECT pattern, mode FROM title_chunking ORDER BY pattern"
                ).fetchall()
                
                if patterns:
                    for pattern, mode in patterns:
                        pcol1, pcol2, pcol3 = st.columns([4, 1, 1])
                        pcol1.write(f"**{pattern}**")
                        pcol2.write(f"`{mode}`")
                        if pcol3.button("🗑️", key=f"del_title_{pattern}", help="Remove"):
                            conn.execute("DELETE FROM title_chunking WHERE pattern=?", (pattern,))
                            conn.commit()
                            st.rerun()
                else:
                    st.info("No title patterns yet. Add one above.")
            else:
                st.warning("title_chunking table not found. Run sync first.")
            
            conn.close()
            
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# SECTION 3b: Ingest Control
# =============================================================================

with st.expander("🔄 Ingest Control", expanded=False):
    import subprocess
    import platform

    def get_ingest_pid():
        """Find running batch_ingest.py process PID. Returns int or None."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ['wmic', 'process', 'where', "commandline like '%batch_ingest%' and name='python.exe'",
                     'get', 'processid', '/format:list'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    if line.startswith("ProcessId="):
                        pid = line.split("=")[1].strip()
                        if pid.isdigit():
                            return int(pid)
            return None
        except Exception:
            return None

    def start_ingest():
        """Start ingest via Task Scheduler."""
        try:
            result = subprocess.run(
                ['schtasks', '/run', '/tn', 'AlexandriaIngest'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def stop_ingest(pid: int):
        """Kill ingest process by PID."""
        try:
            result = subprocess.run(
                ['taskkill', '/PID', str(pid), '/F'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    # --- Status ---
    ingest_pid = get_ingest_pid()
    if ingest_pid:
        st.success(f"🟢 Ingest active (PID {ingest_pid})")
    else:
        st.info("⚪ Ingest idle")

    col1, col2 = st.columns(2)

    with col1:
        if ingest_pid:
            if st.button("⛔ Stop Ingest", type="primary", key="stop_ingest"):
                ok, msg = stop_ingest(ingest_pid)
                if ok:
                    st.success(f"✅ Stopped (PID {ingest_pid}). Refreshing...")
                    import time; time.sleep(1)
                else:
                    st.error(f"Failed: {msg}")
                st.rerun()
        else:
            if st.button("▶️ Start Ingest", type="primary", key="start_ingest"):
                ok, msg = start_ingest()
                if ok:
                    st.success("✅ Started via Task Scheduler. Refreshing...")
                    import time; time.sleep(2)
                else:
                    st.error(f"Failed: {msg}")
                st.rerun()

    with col2:
        if st.button("🔄 Refresh Status", key="refresh_ingest_status"):
            st.rerun()

    # Debug: show raw wmic output if no process found
    if not ingest_pid:
        try:
            result = subprocess.run(
                ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline', '/format:list'],
                capture_output=True, text=True, timeout=5
            )
            python_procs = [l for l in result.stdout.splitlines() if 'batch_ingest' in l.lower()]
            if python_procs:
                st.warning(f"⚠️ wmic sees batch_ingest but PID parse failed: {python_procs}")
        except Exception:
            pass

    # --- Reingest books needing chunking mode change ---
    st.divider()
    st.subheader("♻️ Reingest — Chunking Mismatch")
    st.caption("Books already ingested with wrong chunking mode for current whitelist.")

    try:
        from chunking_policy import get_books_needing_reingest
        db_path = str(ALEXANDRIA_DB) if ALEXANDRIA_DB else ''

        if db_path:
            # Lazy load - only check when button clicked
            if st.button("🔍 Check for Mismatches", key="check_mismatch"):
                with st.spinner("Checking..."):
                    mismatched = get_books_needing_reingest(db_path)
                    st.session_state['mismatched_books'] = mismatched
            
            mismatched = st.session_state.get('mismatched_books', None)
            
            if mismatched is not None:
                if mismatched:
                    st.warning(f"{len(mismatched)} book(s) need reingest.")
                    # Show only first 20 to avoid browser crash
                    import pandas as pd
                    display_data = mismatched[:20]
                    df_mm = pd.DataFrame(display_data)[["title", "author", "current_mode", "should_be"]]
                    df_mm.columns = ["Title", "Author", "Current", "Should Be"]
                    st.dataframe(df_mm, width="stretch", hide_index=True)
                    if len(mismatched) > 20:
                        st.caption(f"Showing first 20 of {len(mismatched)}...")
                else:
                    st.success("✅ All books match current chunking rules.")
        else:
            st.warning("ALEXANDRIA_DB not configured.")
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# SECTION 4: Speaker's Corner
# =============================================================================
with st.expander("🗣️ Speaker's Corner", expanded=True):
    if not qdrant_ok:
        st.error("Qdrant not connected - cannot query")
    elif not OPENROUTER_API_KEY:
        st.warning("OpenRouter API key not configured")
        st.caption("Speaker's Corner requires OpenRouter for answer generation.")
        st.caption("Add OPENROUTER_API_KEY to your .env file.")
    else:
        # Load patterns
        patterns = load_prompt_patterns()

        # Query input
        st.subheader("💬 Your Question")
        query = st.text_area(
            "What do you want to know?",
            placeholder="e.g., What are the key principles of data modeling?",
            key="speaker_query",
            label_visibility="collapsed"
        )

        # Pattern selection (as fragment to avoid full page reruns)
        @st.fragment
        def pattern_selector():
            st.subheader("📝 Response Pattern")

            # Flatten patterns for dropdown
            pattern_options = {"None (just answer)": None}
            for category, items in patterns.items():
                for p in items:
                    display_name = f"{category.title()}: {p['name']}"
                    pattern_options[display_name] = p

            selected_pattern_name = st.selectbox(
                "How should the AI process the results?",
                list(pattern_options.keys()),
                key="speaker_pattern",
                label_visibility="collapsed"
            )
            selected_pattern = pattern_options[selected_pattern_name]

            # Store in session state for use outside fragment
            st.session_state['current_pattern'] = selected_pattern

            # Show pattern details
            if selected_pattern:
                st.info(f"💡 **Use case:** {selected_pattern.get('use_case', '')}")
                st.caption(f"🌡️ Temperature: {selected_pattern.get('temperature', 0.7)}")
                with st.expander("Pattern template"):
                    st.write(selected_pattern.get('template', ''))

        pattern_selector()
        selected_pattern = st.session_state.get('current_pattern')

        # Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            num_results = st.slider("Chunks to retrieve", 3, 15, 5, key="speaker_chunks")
        with col2:
            threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.05, key="speaker_threshold")
        with col3:
            if selected_pattern:
                temperature = st.slider(
                    "Temperature", 0.0, 1.5,
                    selected_pattern.get('temperature', 0.7), 0.1,
                    key="speaker_temp"
                )
            else:
                temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1, key="speaker_temp")

        # Run button
        if st.button("🚀 Ask Alexandria", type="primary", width="stretch"):
            if not query.strip():
                st.warning("Please enter a question")
            else:
                # Build final prompt
                if selected_pattern:
                    final_prompt = f"{query}\n\n---\nResponse instruction: {selected_pattern['template']}"
                else:
                    final_prompt = query

                with st.spinner("Searching knowledge base..."):
                    try:
                        result = perform_rag_query(
                            query=query,
                            collection_name=QDRANT_COLLECTION,
                            limit=num_results,
                            threshold=threshold,
                            host=QDRANT_HOST,
                            port=QDRANT_PORT,
                            generate_llm_answer=True,
                            answer_model=st.session_state.get('selected_model'),
                            openrouter_api_key=OPENROUTER_API_KEY,
                            temperature=temperature,
                            system_prompt=selected_pattern['template'] if selected_pattern else None
                        )

                        # Display answer
                        st.subheader("📜 Answer")
                        if result.answer:
                            st.markdown(result.answer)
                        else:
                            st.warning("No answer generated")

                        # Display sources
                        with st.expander(f"📚 Sources ({len(result.results)} chunks)"):
                            for i, chunk in enumerate(result.results, 1):
                                st.markdown(f"**{i}. {chunk.get('book_title', 'Unknown')}** by {chunk.get('author', 'Unknown')}")
                                st.caption(f"Score: {chunk.get('score', 0):.3f} | {chunk.get('section_name', '')}")
                                st.text(chunk.get('text', '')[:500] + "...")
                                st.divider()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        with st.expander("Details"):
                            import traceback
                            st.code(traceback.format_exc())

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.caption("Alexandria of Temenos • Built with Streamlit • 2026")
