"""Simple fingerprint helpers for caching training artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List


def _hash_file(path: Path) -> bytes:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.digest()


def compute_fingerprint(paths: Iterable[Path], extra: str | None = None) -> str:
    sha = hashlib.sha256()
    normalized: List[Path] = sorted(Path(p) for p in paths)
    for path in normalized:
        sha.update(str(path).encode("utf-8"))
        if path.exists():
            sha.update(str(path.stat().st_size).encode("utf-8"))
            sha.update(str(int(path.stat().st_mtime_ns)).encode("utf-8"))
            sha.update(_hash_file(path))
        else:
            sha.update(b"<missing>")
    if extra:
        sha.update(extra.encode("utf-8"))
    return sha.hexdigest()

