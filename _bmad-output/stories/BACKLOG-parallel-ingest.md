# BACKLOG: Parallel Multi-Host Ingest

**Status:** Backlog
**Priority:** High (after baseline single-worker run completes)
**Type:** Feature

## Summary

Enable parallel book ingestion across multiple hosts with different chunking modes.

## Proposed Architecture

### Workers
| Host | Mode | Hardware | Role |
|------|------|----------|------|
| Minjak | fixed | M4 Pro (MPS) | Fast bulk processing |
| BUCO | semantic | A2000 GPU | Quality processing for whitelisted authors |

### Coordinator
- Runs on SINOVAC (always online)
- Assigns books to workers based on chunking_rules.json
- Tracks progress in SQLite
- Prevents duplicate processing

### Config File
```json
{
  "workers": [
    {"host": "minjak", "mode": "fixed", "qdrant": "192.168.0.151:6333"},
    {"host": "buco", "mode": "semantic", "qdrant": "localhost:6333"}
  ],
  "coordinator": "sinovac",
  "library_path": "/volume1/docker/calibre/alexandria"
}
```

## Implementation Tasks

1. [ ] Create `parallel_coordinator.py` on SINOVAC
2. [ ] Create `worker_client.py` for Minjak/BUCO
3. [ ] Add worker registration/heartbeat
4. [ ] Add job queue (SQLite or Redis)
5. [ ] Add duplicate detection (check ingest_log before assigning)
6. [ ] Add failure recovery (reassign stuck jobs)

## Prerequisites

- [ ] Complete baseline single-worker run for metrics
- [ ] Test Minjak embedding speed (MPS vs CPU)
- [ ] Document error patterns from current run

## Potential Workers

| Host | GPU | CPU | Notes |
|------|-----|-----|-------|
| BUCO | RTX A2000 12GB | - | Primary semantic worker |
| Minjak | M4 Pro (MPS) | - | Primary fixed worker |
| puppet-master | GTX 1060 | i7 7th gen | Lenovo Legion, secondary |
| asus-laptop | GTX ??? | i7 7th gen | Secondary, needs GPU check |

## GUI Requirements

Streamlit dashboard should show:
- [ ] Active workers table (host, GPU, mode, current book, speed)
- [ ] Worker status (online/offline/idle/busy)
- [ ] Real-time progress per worker
- [ ] Total fleet throughput
- [ ] Start/Stop controls per worker
- [ ] Worker registration (add new worker via config)

## Notes

- All workers write to same Qdrant (BUCO localhost:6333)
- Workers connect via Tailscale
- Library mounted read-only on all via SMB
- Worker heartbeat to coordinator every 30s
- GUI polls coordinator for fleet status
