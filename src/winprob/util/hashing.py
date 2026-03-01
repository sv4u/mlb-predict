from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_aggregate_of_files(paths: list[Path]) -> str:
    hashes: list[bytes] = []
    for p in sorted(paths, key=lambda x: str(x)):
        hashes.append(sha256_file(p).encode("ascii"))
    return hashlib.sha256(b"".join(hashes)).hexdigest()
