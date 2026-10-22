import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from unet_nowcast.model import load_model

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model():
    torch.set_num_threads(2)
    return load_model(PROJECT / "artifacts/v1/unet.pth", torch.device("cpu"), (64, 64))


@pytest.fixture
def case(tmp_path):
    payload = json.loads((PROJECT / "config/latam.json").read_text())
    payload["artifacts"] = str(PROJECT / "artifacts/v1")
    payload["domain"] = {"west": -120, "south": 26.6, "east": -113.6, "north": 33}
    payload["model_latitude_order"] = "south_to_north"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    end = datetime(2000, 1, 2, 0, tzinfo=UTC)
    frames = []
    for index in range(12):
        stamp = end - timedelta(hours=11 - index)
        rain = np.zeros((1200, 3600), dtype="<f4")
        patch = np.arange(64, dtype=np.float32)[:, None] / 64 + index / 20
        rain[270:334, 2400:2464] = patch
        rain[270, 2400] = -99  # User's negative-to-zero rule, never NoData.
        frames.append(np.maximum(rain[270:334, 2400:2464], 0).copy())
        with gzip.open(
            inputs / f"gsmap_now.{stamp:%Y%m%d.%H00}.dat.gz", "wb", compresslevel=1
        ) as f:
            f.write(rain.tobytes())
    return {
        "config": config,
        "inputs": inputs,
        "outputs": tmp_path / "outputs",
        "rain": np.stack(frames),
        "run_time": end,
    }
