"""One idempotent polling cycle; systemd owns the scheduling and process lifetime."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .acquisition import DataNotReady, FTPSource, prune_cache
from .artifacts import sha256
from .backend import Backend
from .config import Config
from .pipeline import forecast_directory, require_fresh, run, validate_runtime

LOG = logging.getLogger(__name__)


@contextmanager
def poll_lock(output_dir: Path, product: str, backend: Backend):
    """Advisory lock shared by scheduled invocations using the same output root."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not unlink: replacing the inode would allow overlapping lock holders.
    with (output_dir / f".{backend.slug}-{product}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed_run(
    target: Path, timestamp: datetime, product: str, steps: int | None, backend: Backend
) -> bool:
    if not target.exists():
        return False
    # A malformed/incomplete existing target is an error, never a successful skip.
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    steps = manifest.get("native_output_steps") if steps is None else steps
    if type(steps) is not int or steps < 1:
        raise ValueError(f"Existing run has no valid native horizon: {target}")
    if (
        manifest["model"] != backend.name
        or manifest["status"] != "complete"
        or manifest["run_time"] != timestamp.isoformat()
        or manifest["config"]["product"] != product
        or manifest["output_steps"] != steps
        or len(manifest["outputs"]) != steps
        or {item["lead_hours"] for item in manifest["outputs"]} != set(range(1, steps + 1))
    ):
        raise ValueError(f"Existing run is incomplete or incompatible: {target}")
    for item in manifest["outputs"]:
        path = (target / item["name"]).resolve()
        if path.parent != target.resolve() or sha256(path) != item["sha256"]:
            raise ValueError(f"Existing run failed output integrity checks: {target}")
    return True


def poll_once(args: argparse.Namespace, backend: Backend) -> Path | None:
    validate_runtime(args)
    if args.source != "ftp" or args.run_time is not None:
        raise ValueError("--scheduled requires --source ftp and no --run-time")
    config = Config.load(args.config)
    if "%" in config.remote_directory:
        raise ValueError("--scheduled requires a latest directory, not an archive template")
    with poll_lock(args.output_dir, config.product, backend) as acquired:
        if not acquired:
            LOG.info("BUSY: another scheduled invocation is running")
            return None
        try:
            removed = prune_cache(args.cache_dir, args.cache_ttl_hours)
            if removed:
                LOG.info("Removed %d expired FTP input files from %s", removed, args.cache_dir)
            ftp = FTPSource(config.remote_directory, args.cache_dir, config.domain, config.ftp)
            # A cheap listing precedes model loading and download. Pin the discovered hour
            # so a newly published file cannot silently change this cycle's target.
            latest = ftp.window(config.product, steps=1, run_time=None)[-1].time
            require_fresh(latest, args.max_age_hours)
            target = forecast_directory(args.output_dir, config.product, latest, backend)
            if completed_run(target, latest, config.product, args.steps, backend):
                LOG.info("UP_TO_DATE: %s already completed", latest.isoformat())
                return None
            LOG.info("NEW_DATA: processing %s", latest.isoformat())
            invocation = argparse.Namespace(**vars(args))
            invocation.run_time = latest
            invocation.scheduled = False
            return run(invocation, backend)
        except DataNotReady as exc:
            LOG.info("WAITING: %s; retry on the next timer tick", exc)
            return None
