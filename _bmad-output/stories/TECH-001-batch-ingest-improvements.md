# TECH-001: Batch Ingest Improvements (Retrospective)

**Status:** Done (implemented without workflow — technical debt documented)
**Date:** 2026-03-19
**Type:** Technical Debt / Bug Fix

## Summary

Emergency improvements to batch ingestion pipeline made during live ingest session.

## Changes Made

### 1. Timeout Protection
- **File:** `scripts/batch_ingest.py`
- **Change:** Added 5-minute timeout per book using `concurrent.futures.ThreadPoolExecutor`
- **Reason:** Books were hanging indefinitely on corrupt/malformed files
- **Log:** Timeout books logged to `X:\calibre\alexandria\.qdrant\timeout_books.txt`

### 2. Chunking Rules System
- **File:** `config/chunking_rules.json`
- **Change:** Unified format for author/title patterns with `semantic: true/false` flag
- **Old:** Separate whitelist with implicit semantic=true
- **New:** Single list with explicit mode per pattern

### 3. GUI Enhancements
- **File:** `alexandria_app.py`
- **Changes:**
  - Added "Mode" column to Ingest Log (shows fixed/semantic)
  - Added Chunking Rules Editor section (visual pattern editing)
  - Added Show all / Last N slider for log pagination
  - Fixed UNC paths for SQLite access from Windows SSH

### 4. Documentation
- **File:** `docs/chunking-modes.md`
- **Added:** Timeout protection section, log locations, retry instructions

## Files Modified

| File | Change Type |
|------|-------------|
| `scripts/batch_ingest.py` | Timeout logic, ThreadPoolExecutor |
| `scripts/chunking_policy.py` | Support for new JSON format |
| `config/chunking_rules.json` | New unified format |
| `alexandria_app.py` | GUI improvements |
| `docs/chunking-modes.md` | Documentation |

## Deployment

- All changes deployed to BUCO via SCP
- Scheduled task `AlexandriaIngest` restarted
- Scheduled task `AlexandriaGUI` created for Streamlit

## Lessons Learned

1. Should have created story BEFORE implementing
2. Documentation should be written alongside code, not after
3. BMad workflow ensures nothing is forgotten

## Follow-up Stories

- [ ] TECH-002: Add Start/Stop buttons to GUI
- [ ] TECH-003: Show live ingest progress in GUI (poll log file)
- [ ] TECH-004: Email/Telegram notification on batch complete
