"""Get symbol source code."""

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from ..storage import IndexStore, record_savings, estimate_savings, cost_avoided as _cost_avoided
from ._utils import index_status_to_tool_error, resolve_repo, resolve_fqn


def _make_meta(timing_ms: float, **kwargs) -> dict:
    """Build a _meta envelope dict."""
    meta = {"timing_ms": round(timing_ms, 1)}
    meta.update(kwargs)
    return meta


def _verify_against_git_sha(
    cached_source: str,
    source_root: Optional[str],
    file_path: str,
    line: int,
    end_line: int,
) -> str:
    """Compare cached source against the working-tree git HEAD content (P1.6).

    Returns one of:
    - ``"git_sha_match"``      — the cached source matches the HEAD slice
                                  of the same file (lines line..end_line).
    - ``"git_sha_mismatch"``   — the file exists in HEAD but the slice differs.
    - ``"git_unavailable"``    — source_root unknown, file isn't tracked in
                                  HEAD, or git is unreachable from this env.

    This is an externally-attested verification mode: the comparison target
    comes from git, not from the same cache the symbol's content_hash was
    derived from. The default ``verify_against="cache"`` mode is self-referential
    and only catches incoherent tamper of ``~/.code-index/<repo>/``; this mode
    catches divergence between the cache and the upstream source.
    """
    if not source_root or not file_path:
        return "git_unavailable"
    root = Path(source_root)
    if not (root / ".git").exists() and not (root / ".git").is_file():
        # Not a git working tree (or worktree pointing elsewhere; bail rather
        # than guess).
        return "git_unavailable"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{file_path}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            # Windows stdio-MCP deadlock guard: never inherit the JSON-RPC pipe
            # as the git child's stdin (Git-for-Windows' cmd\git.exe wrapper
            # blocks forever holding the handle, even for commands that don't
            # read stdin). Mirrors the redirect across the other git spawns.
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "git_unavailable"
    if result.returncode != 0:
        # File not in HEAD (untracked, new file, deleted from HEAD, etc.)
        return "git_unavailable"
    head_content = result.stdout
    if not head_content:
        return "git_unavailable"
    head_lines = head_content.split("\n")
    if line < 1 or end_line < line or end_line > len(head_lines):
        # Symbol line range no longer falls within the HEAD file shape; treat
        # as divergence rather than match.
        return "git_sha_mismatch"
    head_slice = "\n".join(head_lines[line - 1:end_line])
    cached_slice = cached_source.rstrip("\n")
    head_slice = head_slice.rstrip("\n")
    return "git_sha_match" if head_slice == cached_slice else "git_sha_mismatch"


def get_symbol_source(
    repo: str,
    symbol_id: Optional[str] = None,
    symbol_ids: Optional[list[str]] = None,
    verify: bool = False,
    context_lines: int = 0,
    storage_path: Optional[str] = None,
    fqn: Optional[str] = None,
    verify_against: str = "cache",
) -> dict:
    """Get full source of one or more symbols by ID.

    Pass symbol_id (string) for one symbol — returns flat symbol object.
    Pass symbol_ids (array) for batch — returns {symbols, errors}.
    Both modes support verify and context_lines.
    Pass fqn (PHP FQN like 'App\\Models\\User') to resolve via PSR-4.
    """
    # FQN resolution: translate PHP FQN → symbol_id
    if fqn and symbol_id is None and symbol_ids is None:
        resolved, fqn_error = resolve_fqn(repo, fqn, storage_path)
        if resolved is None:
            return {"error": fqn_error or f"Could not resolve FQN '{fqn}'."}
        symbol_id = resolved

    # Normalize: some MCP clients send symbol_ids=[] alongside symbol_id when they mean singular mode
    if symbol_id is not None and symbol_ids is not None and len(symbol_ids) == 0:
        symbol_ids = None
    if symbol_id is None and symbol_ids is None:
        return {"error": "Provide symbol_id (string), symbol_ids (array), or fqn (PHP FQN)."}
    if symbol_id is not None and symbol_ids is not None:
        return {"error": "Provide symbol_id or symbol_ids, not both."}

    batch_mode = symbol_ids is not None
    ids = symbol_ids if batch_mode else [symbol_id]

    start = time.perf_counter()
    context_lines = max(0, min(context_lines, 50))

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}

    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)

    if not index:
        return index_status_to_tool_error(store.inspect_index(owner, name))

    symbols_out = []
    errors_out = []
    seen_files: set = set()
    raw_bytes = 0
    response_bytes = 0

    for sid in ids:
        symbol = index.get_symbol(sid)

        if not symbol:
            errors_out.append({"id": sid, "error": f"Symbol not found: {sid}"})
            continue

        source = store.get_symbol_content(owner, name, sid, _index=index)
        content_dir = store._content_dir(owner, name)
        file_full_path = content_dir / symbol["file"]

        context_before = ""
        context_after = ""
        if context_lines > 0 and source and file_full_path.exists():
            try:
                all_lines = file_full_path.read_text(encoding="utf-8", errors="replace").split("\n")
                s_line = symbol["line"] - 1  # 0-indexed
                e_line = symbol["end_line"]   # exclusive
                before_start = max(0, s_line - context_lines)
                after_end = min(len(all_lines), e_line + context_lines)
                if before_start < s_line:
                    context_before = "\n".join(all_lines[before_start:s_line])
                if e_line < after_end:
                    context_after = "\n".join(all_lines[e_line:after_end])
            except Exception:
                pass

        entry = {
            "id": symbol["id"],
            "kind": symbol["kind"],
            "name": symbol["name"],
            "file": symbol["file"],
            "line": symbol["line"],
            "end_line": symbol["end_line"],
            "signature": symbol["signature"],
            "decorators": symbol.get("decorators", []),
            "docstring": symbol.get("docstring", ""),
            "content_hash": symbol.get("content_hash", ""),
            "source": source or "",
        }
        # P1.4: distinguish "empty source" from "no body cached because we're
        # in metadata_only mode" so downstream agents don't treat the empty
        # string as the symbol's actual source.
        if not source:
            try:
                from .. import config as _cfg
                if _cfg.get("cache_mode", "full") == "metadata_only":
                    entry["source_status"] = "metadata_only_mode"
            except Exception:
                pass
        if context_before:
            entry["context_before"] = context_before
        if context_after:
            entry["context_after"] = context_after

        if verify and source:
            actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            stored_hash = symbol.get("content_hash", "")
            entry["content_verified"] = actual_hash == stored_hash if stored_hash else None
            # P1.6: externally-attested mode compares cached source against the
            # working-tree git HEAD slice of the same file. Surfaced alongside
            # the cache-only verification so callers can see both signals.
            if verify_against == "git_sha":
                entry["git_sha_verification"] = _verify_against_git_sha(
                    cached_source=source,
                    source_root=getattr(index, "source_root", None),
                    file_path=symbol["file"],
                    line=symbol["line"],
                    end_line=symbol["end_line"],
                )

        symbols_out.append(entry)

        # Accumulate token savings
        f = symbol["file"]
        if f not in seen_files:
            seen_files.add(f)
            try:
                raw_bytes += os.path.getsize(file_full_path)
            except OSError:
                pass
        response_bytes += symbol.get("byte_length", 0)

    tokens_saved = estimate_savings(raw_bytes, response_bytes)
    total_saved = record_savings(tokens_saved, tool_name="get_symbol_source")
    elapsed = (time.perf_counter() - start) * 1000
    meta = _make_meta(elapsed, tokens_saved=tokens_saved, total_tokens_saved=total_saved,
                      **_cost_avoided(tokens_saved, total_saved))

    from ..retrieval.freshness import FreshnessProbe as _FreshnessProbe
    _probe = _FreshnessProbe(
        source_root=getattr(index, "source_root", "") or None,
        indexed_at=getattr(index, "indexed_at", ""),
        index_sha=getattr(index, "git_head", None),
        file_mtimes=getattr(index, "file_mtimes", None),
    )
    _probe.annotate(symbols_out)

    # Phase 2: runtime confidence — zero-cost no-op when no traces ingested.
    from ..runtime.confidence import attach_runtime_confidence as _attach_runtime
    _runtime_summary = _attach_runtime(
        symbols_out,
        str(store._sqlite._db_path(owner, name)),
        id_field="id",
    )

    if batch_mode:
        meta["symbol_count"] = len(symbols_out)
        meta["freshness"] = _probe.summary(symbols_out)
        if _runtime_summary:
            meta["runtime_freshness"] = _runtime_summary
        return {"symbols": symbols_out, "errors": errors_out, "_meta": meta}

    # Single mode: flat object or error
    if errors_out:
        return {"error": errors_out[0]["error"]}
    result = symbols_out[0]
    meta["hint"] = "Use get_context_bundle(symbol_id) to retrieve source + imports in one call"
    meta["freshness"] = _probe.summary(symbols_out)
    if _runtime_summary:
        meta["runtime_freshness"] = _runtime_summary
    result["_meta"] = meta
    return result
