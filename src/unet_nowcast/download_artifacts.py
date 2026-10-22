"""Download a versioned model bundle from public HTTP(S) using pinned local hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SIZE = 1024 * 1024


def default_bundle() -> Path:
    source_bundle = PROJECT_ROOT / "artifacts" / "v1"
    if (source_bundle / "manifest.json").is_file():
        return source_bundle
    return Path.cwd() / "artifacts" / "v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_files(manifest: dict) -> list[tuple[str, str]]:
    """Accept the GSMaP and IRENE manifest formats without trusting remote metadata."""
    if "files" in manifest:
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported GSMaP artifact manifest version")
        items = [manifest["files"][role] for role in ("checkpoint", "mean", "std")]
        files = [(item["name"], item["sha256"]) for item in items]
    elif manifest.get("version") == 1:
        files = [(manifest["model_file"], manifest["sha256"])]
    else:
        raise ValueError("Unsupported artifact manifest format")
    names = set()
    for name, digest in files:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in names
        ):
            raise ValueError(f"Unsafe or duplicate artifact filename: {name!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for artifact {name}")
        names.add(name)
    return files


def file_url(base_url: str, name: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Artifact base URL must be a public HTTP(S) directory URL")
    path = parsed.path.rstrip("/") + "/" + quote(name, safe="")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def download_one(url: str, target: Path, expected_hash: str, attempts: int, timeout: float) -> str:
    if target.exists() and sha256(target) == expected_hash:
        return "already verified"
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".part",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                digest = hashlib.sha256()
                with urlopen(url, timeout=timeout) as response:
                    for block in iter(lambda: response.read(BLOCK_SIZE), b""):
                        stream.write(block)
                        digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {target.name}")
            os.replace(temporary, target)
            return "downloaded and verified"
        except (OSError, ValueError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(
                    f"Could not download {target.name} after {attempts} attempt(s): {exc}"
                ) from exc
            time.sleep(min(2**attempt, 30))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Download missing model artifacts and verify them against the local manifest"
    )
    command.add_argument(
        "--bundle",
        type=Path,
        default=default_bundle(),
        help="Local bundle directory containing a trusted manifest.json",
    )
    command.add_argument(
        "--base-url", help="Override the download_base_url recorded in the local manifest"
    )
    command.add_argument("--attempts", type=int, default=3)
    command.add_argument("--timeout", type=float, default=60.0)
    command.add_argument(
        "--list", action="store_true", help="Print expected URLs and hashes without downloading"
    )
    return command


def main(argv: list[str] | None = None) -> int:
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.attempts < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--attempts and --timeout must be positive")
        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        files = bundle_files(manifest)
        base_url = args.base_url or manifest.get("download_base_url")
        if not base_url:
            raise ValueError("No download URL configured; provide --base-url")
        urls = [(name, digest, file_url(base_url, name)) for name, digest in files]
        if args.list:
            for name, digest, url in urls:
                print(f"{url}  SHA-256={digest}  -> {args.bundle / name}")
            return 0
        for name, digest, url in urls:
            result = download_one(url, args.bundle / name, digest, args.attempts, args.timeout)
            print(f"{name}: {result}")
        print(f"Bundle ready: {args.bundle}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Artifact download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
