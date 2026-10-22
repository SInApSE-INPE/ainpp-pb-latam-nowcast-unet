"""Acquire an exact consecutive UTC window from local files or JAXA FTP."""

from __future__ import annotations

import ftplib
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from .config import Domain, FTPSettings
from .processing import read_gsmap

LOG = logging.getLogger(__name__)


class DataNotReady(ValueError):
    """A complete, current observation window is not available yet."""


class IncompleteDownload(OSError):
    """The remote object may still be uploading; retry the transfer."""


@dataclass(frozen=True)
class Observation:
    time: datetime
    name: str
    location: str


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Time must include a timezone, e.g. 2026-10-19T12:00:00Z")
    parsed = parsed.astimezone(UTC)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("GSMaP run time must be on an exact UTC hour")
    return parsed


def catalog(locations: list[str], product: str) -> list[Observation]:
    interval = r"(?:_(?P<end>\d{2})00)?" if product == "gsmap_now" else ""
    pattern = re.compile(
        rf"{re.escape(product)}\.(?P<date>\d{{8}})\.(?P<hour>\d{{2}})00" + interval + r"\.dat\.gz"
    )
    observations = []
    for location in locations:
        name = PurePosixPath(location).name
        match = pattern.fullmatch(name)
        if not match:
            continue
        timestamp = datetime.strptime(match["date"] + match["hour"], "%Y%m%d%H").replace(tzinfo=UTC)
        if product == "gsmap_now" and match["end"] is not None:
            if int(match["end"]) != (timestamp.hour + 1) % 24:
                raise ValueError(f"Expected a one-hour NOW interval: {name}")
        observations.append(Observation(timestamp, name, location))
    return observations


def select_window(
    observations: list[Observation], steps: int, run_time: datetime | None = None
) -> list[Observation]:
    if steps < 1:
        raise ValueError("input_steps must be positive")
    if not observations:
        raise DataNotReady("No hourly observations available yet")
    run_time = run_time or max(item.time for item in observations)
    by_time: dict[datetime, Observation] = {}
    for item in observations:
        if item.time in by_time:
            raise ValueError(f"Ambiguous duplicate observation for {item.time.isoformat()}")
        by_time[item.time] = item
    required = [run_time - timedelta(hours=i) for i in reversed(range(steps))]
    missing = [t.isoformat() for t in required if t not in by_time]
    if missing:
        raise DataNotReady(f"Incomplete hourly window; missing: {', '.join(missing)}")
    return [by_time[t] for t in required]


def local_window(directory: Path, product: str, steps: int, run_time: datetime | None):
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    return select_window(
        catalog([str(p.resolve()) for p in directory.rglob("*.dat.gz")], product), steps, run_time
    )


def prune_cache(cache: Path, ttl_hours: float, now: float | None = None) -> int:
    """Remove expired downloaded inputs from managed hash directories only."""
    if not math.isfinite(ttl_hours) or ttl_hours <= 0:
        raise ValueError("cache TTL must be finite and positive")
    if not cache.exists():
        return 0
    if cache.is_symlink() or not cache.is_dir():
        raise ValueError(f"Cache path must be a directory, not a symlink: {cache}")
    cutoff = (time.time() if now is None else now) - ttl_hours * 3600
    removed = 0
    for namespace in cache.iterdir():
        if (
            namespace.is_symlink()
            or not namespace.is_dir()
            or not re.fullmatch(r"[0-9a-f]{16}", namespace.name)
        ):
            continue
        for path in namespace.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.name.endswith((".dat.gz", ".part"))
            ):
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        try:
            namespace.rmdir()
        except OSError:
            pass
    return removed


class FTPSource:
    def __init__(self, directory: str, cache: Path, domain: Domain, settings: FTPSettings):
        self.host = os.environ.get("FTP_JAXA_HOST", "hokusai.eorc.jaxa.jp")
        self.user = os.environ.get("FTP_JAXA_USER")
        self.password = os.environ.get("FTP_JAXA_PASSWORD")
        if not self.user or not self.password:
            raise ValueError("Set FTP_JAXA_USER and FTP_JAXA_PASSWORD for FTP acquisition")
        self.directory, self.cache, self.domain = directory, cache, domain
        self.settings = settings

    def _request(self, operation):
        for attempt in range(self.settings.attempts):
            try:
                with ftplib.FTP(timeout=self.settings.timeout_seconds) as ftp:
                    ftp.connect(self.host)
                    ftp.login(self.user, self.password)
                    return operation(ftp)
            except ftplib.all_errors:
                if attempt + 1 == self.settings.attempts:
                    # Do not expose server responses or credentials in logs.
                    raise ConnectionError(
                        f"FTP request failed after {self.settings.attempts} attempts"
                    ) from None
                delay = min(
                    self.settings.retry_delay_seconds * 2**attempt,
                    self.settings.max_retry_delay_seconds,
                )
                LOG.warning(
                    "FTP request failed; retry %s/%s in %ss",
                    attempt + 2,
                    self.settings.attempts,
                    delay,
                )
                time.sleep(delay)

    def list_observations(self, product: str, steps: int, run_time: datetime | None):
        if "%" in self.directory:
            if run_time is None:
                raise ValueError("Archive directory templates require --run-time")
            directories = sorted(
                {(run_time - timedelta(hours=i)).strftime(self.directory) for i in range(steps)}
            )
        else:
            directories = [self.directory]
        observations = []
        for directory in directories:

            def listing(ftp, directory=directory):
                ftp.cwd(directory)
                return ftp.nlst()

            names = self._request(listing)
            locations = [str(PurePosixPath(directory) / PurePosixPath(n).name) for n in names]
            observations.extend(catalog(locations, product))
        return observations

    def window(self, product: str, steps: int, run_time: datetime | None):
        return select_window(self.list_observations(product, steps, run_time), steps, run_time)

    def download(self, item: Observation) -> Path:
        # Include host and remote directory identity to separate provider/product versions.
        import hashlib

        namespace = hashlib.sha256(f"{self.host}:{item.location}".encode()).hexdigest()[:16]
        target = self.cache / namespace / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                read_gsmap(target, self.domain)
                return target
            except (OSError, EOFError, ValueError):
                LOG.warning("Invalid cached file; downloading %s again", item.name)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:

            def retrieve(ftp):
                with temporary.open("wb") as stream:
                    ftp.retrbinary(f"RETR {item.location}", stream.write)
                try:
                    read_gsmap(temporary, self.domain)
                except (OSError, EOFError, ValueError) as exc:
                    raise IncompleteDownload(f"Incomplete GSMaP transfer: {item.name}") from exc

            self._request(retrieve)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
