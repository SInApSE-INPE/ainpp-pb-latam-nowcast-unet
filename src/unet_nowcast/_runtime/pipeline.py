"""CLI orchestration: load, acquire, preprocess, infer and atomically publish."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import torch

from . import __version__
from .acquisition import DataNotReady, FTPSource, local_window, parse_time, prune_cache
from .artifacts import sha256, verify_bundle
from .backend import Backend
from .config import Config
from .export import write_geotiff
from .formats import CRS, PRECIPITATION_UNITS
from .processing import Normalization, prepare_inputs

LOG = logging.getLogger(__name__)


def parser(backend: Backend) -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description=f"{backend.name} hourly precipitation to GeoTIFF (UTC)"
    )
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--source", choices=("local", "ftp"), default="local")
    cli.add_argument("--input-dir", type=Path, help="Local GSMaP .dat.gz files (recursive)")
    cli.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / f"nowcasting-{os.getuid()}" / backend.slug,
        help="Temporary directory for FTP inputs; local inputs are never removed",
    )
    cli.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=24.0,
        help="Delete expired FTP inputs on each run or scheduled poll (default: 24 hours)",
    )
    cli.add_argument("--output-dir", type=Path, default=Path("outputs"))
    cli.add_argument(
        "--run-time", type=parse_time, help="Timestamp of LAST input, timezone required"
    )
    cli.add_argument(
        "--steps",
        type=int,
        default=backend.default_steps,
        help="Forecast hours; defaults to the native model horizon",
    )
    cli.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cli.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    cli.add_argument("--max-age-hours", type=float, default=12, help="Latest FTP window age limit")
    cli.add_argument("--verbose", action="store_true")
    cli.add_argument(
        "--scheduled",
        action="store_true",
        help="One live FTP poll: skip completed runs, lock concurrent polls, wait for missing data",
    )
    return cli


def validate_runtime(args: argparse.Namespace) -> None:
    if (args.steps is not None and args.steps < 1) or args.threads < 1:
        raise ValueError("steps and threads must be positive")
    if not math.isfinite(args.max_age_hours) or args.max_age_hours <= 0:
        raise ValueError("max-age-hours must be finite and positive")
    if not math.isfinite(args.cache_ttl_hours) or args.cache_ttl_hours <= 0:
        raise ValueError("cache-ttl-hours must be finite and positive")


def forecast_directory(root: Path, product: str, run_time: datetime, backend: Backend) -> Path:
    return root / backend.slug / product / run_time.strftime("%Y/%m/%d/%H%MZ")


def require_fresh(run_time: datetime, max_age_hours: float) -> None:
    age = (datetime.now(UTC) - run_time).total_seconds() / 3600
    if age < 0 or age > max_age_hours:
        raise DataNotReady(f"Latest FTP observation is stale or in the future (age={age:.1f}h)")


def run(args: argparse.Namespace, backend: Backend) -> Path:
    validate_runtime(args)
    if args.source == "ftp":
        removed = prune_cache(args.cache_dir, args.cache_ttl_hours)
        if removed:
            LOG.info("Removed %d expired FTP input files from %s", removed, args.cache_dir)
    config = Config.load(args.config)
    bundle, artifacts = verify_bundle(config.artifacts)
    LOG.info("Verified model bundle %s", bundle["version"])
    reference_product = bundle["reference_input_product"]
    if config.product != reference_product:
        LOG.warning(
            "%s inputs differ from %s in the supplied inference reference",
            config.product,
            reference_product,
        )
    if bundle.get("training_latitude_order") is None:
        LOG.warning(
            "Training latitude order is unverified; using configured %s",
            config.model_latitude_order,
        )
    elif bundle["training_latitude_order"] != config.model_latitude_order:
        raise ValueError("Configured latitude order differs from model bundle")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_num_threads(args.threads)
    model = backend.load_model(artifacts["checkpoint"], device, config.domain.shape)
    steps = args.steps if args.steps is not None else getattr(model, "output_steps", None)
    if steps is None or steps < 1:
        raise ValueError("Model must define its native output horizon or require --steps")
    if hasattr(model, "output_steps") and steps > model.output_steps:
        raise ValueError(f"Requested {steps} hours exceeds native horizon {model.output_steps}")
    norm = Normalization.load(artifacts["mean"], artifacts["std"])
    LOG.info("Model ready on %s; %s input hours", device, model.input_steps)
    ftp = None
    if args.source == "local":
        if args.input_dir is None:
            raise ValueError("--source local requires --input-dir")
        observations = local_window(
            args.input_dir, config.product, model.input_steps, args.run_time
        )
    else:
        ftp = FTPSource(config.remote_directory, args.cache_dir, config.domain, config.ftp)
        observations = ftp.window(config.product, model.input_steps, args.run_time)
        if args.run_time is None:
            require_fresh(observations[-1].time, args.max_age_hours)
    run_time = observations[-1].time
    target = forecast_directory(args.output_dir, config.product, run_time, backend)
    if target.exists():
        raise FileExistsError(f"Run already exists: {target}; use a different output directory")
    paths = [ftp.download(item) if ftp else Path(item.location) for item in observations]
    LOG.info("Acquired %s consecutive inputs ending at %s", len(paths), run_time.isoformat())
    inputs, missing_mask, fractions = prepare_inputs(paths, config, norm)
    tensor = torch.from_numpy(inputs).unsqueeze(0).to(device)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "run_time": run_time.isoformat(),
        "model": backend.name,
        "bundle_version": bundle["version"],
        "artifacts": bundle["files"],
        "config": {**asdict(config), "artifacts": str(config.artifacts)},
        "normalization": asdict(norm),
        "training_latitude_order_verified": bundle.get("training_latitude_order") is not None,
        "source": args.source,
        "ftp_host": ftp.host if ftp else None,
        "cache_ttl_hours": args.cache_ttl_hours if ftp else None,
        "units": PRECIPITATION_UNITS,
        "crs": CRS,
        "resolution_degrees": config.domain.resolution,
        "device": str(device),
        "versions": {
            backend.slug: backend.version,
            "nowcasting-runtime": __version__,
            **backend.library_versions,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
        "input_steps": model.input_steps,
        "output_steps": steps,
        "native_output_steps": getattr(model, "output_steps", None),
        "masked_output_fraction": float(missing_mask.mean()),
        "inputs": [
            {
                "name": item.name,
                "time": item.time.isoformat(),
                "source": item.location,
                "sha256": sha256(path),
                "missing_fraction": fraction,
            }
            for item, path, fraction in zip(observations, paths, fractions, strict=True)
        ],
        "outputs": [],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish the entire run only after every GeoTIFF and its manifest have been written.
    with tempfile.TemporaryDirectory(prefix=f".{backend.slug}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        for lead, prediction in enumerate(backend.forecast_steps(model, tensor, steps), start=1):
            if lead > steps or tuple(prediction.shape) != (1, 1, *config.domain.shape):
                raise ValueError("Model returned an invalid forecast shape or extra lead")
            rain = norm.inverse(prediction[0, 0].cpu().numpy())
            if config.model_latitude_order == "south_to_north":
                rain = rain[::-1, :]
            valid_time = run_time + timedelta(hours=lead)
            name = (
                f"{backend.slug}_{run_time:%Y%m%dT%H%MZ}_f{lead:03d}_{valid_time:%Y%m%dT%H%MZ}.tif"
            )
            tags = {
                "model": backend.name,
                "model_version": bundle["version"],
                "checkpoint_sha256": bundle["files"]["checkpoint"]["sha256"],
                "run_time": run_time.isoformat(),
                "valid_time": valid_time.isoformat(),
                "lead_hours": lead,
                "input_product": config.product,
                "model_latitude_order": config.model_latitude_order,
                "missing_policy": config.missing_policy,
            }
            write_geotiff(staging / name, rain, missing_mask, config.domain, tags)
            metadata["outputs"].append(
                {
                    "name": name,
                    "valid_time": valid_time.isoformat(),
                    "lead_hours": lead,
                    "sha256": sha256(staging / name),
                }
            )
            LOG.info("Generated lead +%sh: %s", lead, name)
        if len(metadata["outputs"]) != steps:
            raise ValueError("Model returned fewer leads than requested")
        (staging / "manifest.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    LOG.info("Published complete run: %s", target)
    return target


def main(backend: Backend, argv=None) -> int:
    args = parser(backend).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.scheduled:
            from .scheduler import poll_once

            poll_once(args, backend)
        else:
            run(args, backend)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError) as exc:
        LOG.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        return 1
    return 0
