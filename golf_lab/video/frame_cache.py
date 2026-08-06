"""A tiny in-memory LRU cache for decoded frames.

Deliberately simple: decoded frames are large, so the cache holds a small,
bounded number of them. Cache keys are deterministic (see ``frame_cache_key``)
so the same video + settings always produce the same key, which matters once
pose results and renders are cached on disk in later milestones.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional


def file_fingerprint(path: Path) -> str:
    """Cheap, stable identity for a file: size + mtime + name.

    Hashing multi-gigabyte video content on every page load would be far too
    slow; size and mtime are sufficient to detect the realistic case of the
    file being replaced.
    """
    path = Path(path)
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def frame_cache_key(path: Path, frame_index: int, **settings: Any) -> str:
    """Deterministic cache key for one decoded frame under given settings."""
    settings_part = "|".join(f"{k}={settings[k]}" for k in sorted(settings))
    return f"{file_fingerprint(path)}|{frame_index}|{settings_part}"


class FrameCache:
    """Bounded LRU cache mapping cache key -> decoded frame array."""

    def __init__(self, max_items: int = 24) -> None:
        if max_items < 1:
            raise ValueError("max_items must be at least 1")
        self.max_items = max_items
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store
