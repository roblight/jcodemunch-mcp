"""Session state persistence across server restarts.

This module provides SessionState class for persisting and restoring:
- Session journal (file reads, searches, edits)
- Search result cache
- Negative evidence log

Storage location: ~/.code-index/_session_state.json
"""
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional


class SessionState:
    """Persist and restore session state across server restarts."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize session state with storage path.
        
        Args:
            base_path: Storage directory. Defaults to CODE_INDEX_PATH env var.
        """
        if base_path is None:
            base_path = os.environ.get("CODE_INDEX_PATH", str(Path.home() / ".code-index"))
        self._path = Path(base_path) / "_session_state.json"
        self._lock = threading.Lock()
        self._flush_counter = 0

    def save(
        self,
        journal: Any,
        search_cache: OrderedDict,
        max_queries: int = 50,
        negative_evidence_log: Optional[list] = None,
    ) -> None:
        """Serialize journal + top search cache entries to disk.
        
        Args:
            journal: SessionJournal instance
            search_cache: OrderedDict of cached search results
            max_queries: Maximum cache entries to save (by hit_count)
            negative_evidence_log: Optional list of negative evidence entries
        """
        with self._lock:
            # Get journal context
            ctx = journal.get_context(max_files=1000, max_queries=1000, max_edits=1000)
            
            # Build journal data
            journal_data = {
                "files_accessed": {
                    f["file"]: {"reads": f["reads"], "last_tool": f["last_tool"]}
                    for f in ctx["files_accessed"]
                },
                "queries": {
                    q["query"]: {"count": q["count"], "result_count": q["result_count"]}
                    for q in ctx["recent_searches"]
                },
                "files_edited": {
                    e["file"]: {"edits": e["edits"]}
                    for e in ctx["files_edited"]
                },
            }
            
            # Build search cache data (sorted by hit_count, capped)
            cache_entries = []
            for key, value in search_cache.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    cache_entries.append({
                        "repo": key[0] if key[0] else "",
                        "indexed_at": key[1] if len(key) > 1 else "",
                        "query": key[2] if len(key) > 2 else "",
                        "key": list(key),  # Store full key for reconstruction
                        "result": value,
                        "hit_count": value.get("_hit_count", 1),
                    })
            
            # Sort by hit_count descending, take top N
            cache_entries.sort(key=lambda e: e["hit_count"], reverse=True)
            cache_entries = cache_entries[:max_queries]
            
            # Build index snapshots (indexed_at per repo)
            index_snapshots = {}
            for entry in cache_entries:
                repo = entry.get("repo", "")
                indexed_at = entry.get("indexed_at", "")
                if repo and indexed_at:
                    index_snapshots[repo] = indexed_at
            
            # Build state object
            state = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "index_snapshots": index_snapshots,
                "journal": journal_data,
                "search_cache": cache_entries,
                "negative_evidence_log": negative_evidence_log or [],
            }
            
            # Write atomically
            temp_path = self._path.with_suffix(".json.tmp")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp_path.replace(self._path)
            
            self._flush_counter += 1

    def load(self, max_age_minutes: int = 60) -> Optional[dict]:
        """Load saved state if fresh enough.
        
        Args:
            max_age_minutes: Maximum age in minutes before discarding
            
        Returns:
            State dict or None if stale/missing
        """
        with self._lock:
            if not self._path.exists():
                return None
            
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                return None
            
            # Check age
            saved_at_str = data.get("saved_at", "")
            if not saved_at_str:
                return None
            
            try:
                saved_at = datetime.fromisoformat(saved_at_str.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - saved_at
                if age > timedelta(minutes=max_age_minutes):
                    return None
            except Exception:
                return None
            
            return data

    def restore_journal(self, journal: Any, data: dict) -> int:
        """Populate a journal from saved data.
        
        Args:
            journal: SessionJournal instance to populate
            data: Loaded state dict
            
        Returns:
            Count of restored entries
        """
        if not data or "journal" not in data:
            return 0
        
        journal_data = data["journal"]
        count = 0
        
        # Restore file reads
        for file_path, entry in journal_data.get("files_accessed", {}).items():
            for _ in range(entry.get("reads", 1)):
                journal.record_read(file_path, entry.get("last_tool", "unknown"))
                count += 1
        
        # Restore searches (replay count times to preserve search frequency)
        for query, entry in journal_data.get("queries", {}).items():
            for _ in range(entry.get("count", 1)):
                journal.record_search(query, entry.get("result_count", 0))
                count += 1

        # Restore edits (replay edit count)
        for file_path, entry in journal_data.get("files_edited", {}).items():
            for _ in range(entry.get("edits", 1) if isinstance(entry, dict) else 1):
                journal.record_edit(file_path)
                count += 1
        
        return count

    def restore_search_cache(
        self,
        cache: OrderedDict,
        data: dict,
        current_indexes: dict,
    ) -> int:
        """Restore search cache entries, skipping any where index has changed.
        
        Args:
            cache: OrderedDict to populate
            data: Loaded state dict
            current_indexes: {repo: indexed_at} for staleness check
            
        Returns:
            Count of restored entries
        """
        if not data or "search_cache" not in data:
            return 0
        
        count = 0
        for entry in data["search_cache"]:
            repo = entry.get("repo", "")
            saved_indexed_at = entry.get("indexed_at", "")
            
            # Check if index has changed
            current_indexed_at = current_indexes.get(repo, "")
            if current_indexed_at and current_indexed_at != saved_indexed_at:
                # Index changed, skip this entry
                continue
            
            # Reconstruct cache key
            key = tuple(entry.get("key", []))
            if not key:
                continue
            
            # Restore entry
            cache[key] = entry.get("result", {})
            count += 1
        
        return count

    def clear(self) -> None:
        """Delete saved state file."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()


# Singleton for global access
_session_state: Optional[SessionState] = None
_session_state_lock = threading.Lock()


def get_session_state() -> SessionState:
    """Get the singleton SessionState instance."""
    global _session_state
    with _session_state_lock:
        if _session_state is None:
            _session_state = SessionState()
        return _session_state