"""Binary decoding, spatial orientation and the training normalization contract."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config, Domain
from .formats import GSMAP_BYTES, GSMAP_DTYPE, GSMAP_NORTH, GSMAP_SHAPE, GSMAP_WEST


def read_gsmap(path: Path, domain: Domain) -> np.ndarray:
    # Reading the complete bounded stream also checks gzip CRC/truncation.
    with gzip.open(path, "rb") as stream:
        raw = stream.read(GSMAP_BYTES + 1)
    if len(raw) != GSMAP_BYTES:
        raise ValueError(f"Invalid GSMaP size in {path.name}: {len(raw)} bytes")
    data = np.frombuffer(raw, dtype=GSMAP_DTYPE).reshape(GSMAP_SHAPE)
    # JAXA starts at 0.05 E, 59.95 N. Shift longitude; retain north-first rows.
    data = np.roll(data, GSMAP_SHAPE[1] // 2, axis=1)
    row0 = round((GSMAP_NORTH - domain.north) / domain.resolution)
    col0 = round((domain.west - GSMAP_WEST) / domain.resolution)
    height, width = domain.shape
    return data[row0 : row0 + height, col0 : col0 + width].copy()


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("Normalization requires finite mean and positive finite std")

    @classmethod
    def load(cls, mean: Path, std: Path) -> Normalization:
        return cls(
            float(np.load(mean, allow_pickle=False).reshape(())),
            float(np.load(std, allow_pickle=False).reshape(())),
        )

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            rain = np.expm1(normalized.astype(np.float32) * self.std + self.mean)
        if not np.isfinite(rain).all():
            raise ValueError("Non-finite precipitation after inverse normalization")
        return np.maximum(rain, 0).astype(np.float32)


def prepare_inputs(paths: list[Path], config: Config, norm: Normalization):
    frames, masks, fractions = [], [], []
    for path in paths:
        rain = read_gsmap(path, config.domain)
        missing = ~np.isfinite(rain)
        # Clip physical rainfall before log1p; normalized values may still be negative.
        np.maximum(rain, 0, out=rain)
        fraction = float(missing.mean())
        if fraction > config.max_missing_fraction:
            raise ValueError(f"Missing fraction {fraction:.3%} exceeds limit in {path.name}")
        if missing.any() and config.missing_policy == "error":
            raise ValueError(f"Missing precipitation in {path.name}; choose an explicit policy")
        rain[missing] = 0
        normalized = (np.log1p(rain) - norm.mean) / norm.std
        if config.missing_policy == "normalized_zero":
            normalized[missing] = 0
        if not np.isfinite(normalized).all():
            raise ValueError(f"Non-finite normalized input in {path.name}")
        frames.append(normalized)
        masks.append(missing)
        fractions.append(fraction)
    inputs = np.stack(frames).astype(np.float32)
    if config.model_latitude_order == "south_to_north":
        inputs = inputs[:, ::-1, :]
    # Mask all pixels with missing history in every output; keep this mask north-first.
    return np.ascontiguousarray(inputs), np.any(masks, axis=0), fractions
