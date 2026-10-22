"""Versioned model bundle with checksums for weights and normalization parameters."""

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(directory: Path) -> tuple[dict, dict[str, Path]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported artifact manifest version")
    paths = {}
    for role in ("checkpoint", "mean", "std"):
        item = manifest["files"][role]
        path = (directory / item["name"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Artifact files must be inside their bundle directory")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {role}")
        paths[role] = path
    return manifest, paths
