"""Validated configuration; all paths are relative to the configuration file."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .formats import GSMAP_EAST, GSMAP_NORTH, GSMAP_RESOLUTION, GSMAP_SOUTH, GSMAP_WEST


@dataclass(frozen=True)
class Domain:
    west: float
    south: float
    east: float
    north: float

    @property
    def resolution(self) -> float:
        return GSMAP_RESOLUTION

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Domain values must be finite")
        if not (
            GSMAP_WEST <= self.west < self.east <= GSMAP_EAST
            and GSMAP_SOUTH <= self.south < self.north <= GSMAP_NORTH
        ):
            raise ValueError("Domain outside the GSMaP binary grid")
        for value in (self.west, self.south, self.east, self.north):
            if not math.isclose(value / self.resolution, round(value / self.resolution)):
                raise ValueError("Domain edges must align with the GSMaP grid")

    @property
    def shape(self) -> tuple[int, int]:
        return (
            round((self.north - self.south) / self.resolution),
            round((self.east - self.west) / self.resolution),
        )


@dataclass(frozen=True)
class FTPSettings:
    timeout_seconds: float
    attempts: int
    retry_delay_seconds: float
    max_retry_delay_seconds: float

    def __post_init__(self):
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("FTP attempts must be a positive integer")
        for value in (self.timeout_seconds, self.retry_delay_seconds, self.max_retry_delay_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("FTP timeouts and retry delays must be finite and positive")
        if self.max_retry_delay_seconds < self.retry_delay_seconds:
            raise ValueError("FTP maximum retry delay must not be less than initial delay")


@dataclass(frozen=True)
class Config:
    artifacts: Path
    domain: Domain
    model_latitude_order: str
    product: str
    remote_directory: str
    missing_policy: str
    max_missing_fraction: float
    ftp: FTPSettings

    def __post_init__(self):
        if self.model_latitude_order not in {"north_to_south", "south_to_north"}:
            raise ValueError("model_latitude_order must be north_to_south or south_to_north")
        if self.product not in {"gsmap_nrt", "gsmap_now"}:
            raise ValueError("Only gsmap_nrt and gsmap_now hourly rain rates are supported")
        if self.missing_policy not in {"error", "normalized_zero", "zero_rain"}:
            raise ValueError("Unknown missing_policy")
        if not 0 <= self.max_missing_fraction < 1:
            raise ValueError("max_missing_fraction must be in [0, 1)")

    @classmethod
    def load(cls, path: Path) -> Config:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"] = (path.resolve().parent / payload["artifacts"]).resolve()
        payload["domain"] = Domain(**payload["domain"])
        payload["ftp"] = FTPSettings(**payload["ftp"])
        return cls(**payload)
