import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from unet_nowcast._runtime import pipeline, scheduler
from unet_nowcast._runtime.artifacts import sha256, verify_bundle
from unet_nowcast.entrypoint import BACKEND, parser, run
from unet_nowcast.model import forecast_steps, load_model
from unet_nowcast.processing import Normalization

PROJECT = Path(__file__).resolve().parents[1]


def test_reference_equivalence_and_input_contract(model):
    spec = importlib.util.spec_from_file_location("reference_unet", PROJECT / "reference/unet.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    original = reference.PrecipitationForecaster(
        input_timesteps=model.input_steps,
        output_timesteps=model.output_steps,
        input_channels=1,
        hidden_channels=[64, 128, 256, 512, 1024],
        kernel_size=7,
    ).eval()
    original.load_state_dict(model.state_dict(), strict=True)
    inputs = torch.linspace(-1, 1, 12 * 64 * 64).reshape(1, 12, 64, 64)
    with torch.inference_mode():
        expected = original(inputs.unsqueeze(2))
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs.unsqueeze(2)), actual, rtol=0, atol=0)
    assert actual.shape == (1, 6, 1, 64, 64)
    assert actual.min() >= 0
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(1, 6, 64, 64))


def test_direct_prediction_uses_one_forward_and_keeps_lead_order():
    class CountingModel(torch.nn.Module):
        output_steps = 6
        img_size = (2, 3)
        calls = 0

        def forward(self, x):
            self.calls += 1
            return torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1, 1).expand(1, 6, 1, 2, 3)

    model = CountingModel()
    leads = list(forecast_steps(model, torch.zeros(1, 12, 2, 3), 3))
    assert model.calls == 1
    assert [float(lead[0, 0, 0, 0]) for lead in leads] == [0, 1, 2]
    with pytest.raises(ValueError, match="native horizon"):
        list(forecast_steps(model, torch.zeros(1, 12, 2, 3), 7))
    assert model.calls == 1


def test_state_loading_rejects_incompatible_weights(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"wrong.weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="supported U-Net"):
        load_model(path, torch.device("cpu"), (64, 64))


def test_bundle_contains_only_inference_weights():
    manifest, paths = verify_bundle(PROJECT / "artifacts/v1")
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    assert all(torch.is_tensor(v) for v in state.values())
    assert state["unet.inc.double_conv.0.weight"].shape[1] == 12
    assert state["unet.outc.weight"].shape[0] == 6
    assert manifest["source_checkpoint"]["sha256"] == sha256(PROJECT / "reference/best_model.pth")


def args_for(case):
    return parser().parse_args(
        [
            "--config",
            str(case["config"]),
            "--source",
            "local",
            "--input-dir",
            str(case["inputs"]),
            "--output-dir",
            str(case["outputs"]),
            "--device",
            "cpu",
            "--threads",
            "2",
        ]
    )


def test_end_to_end_native_horizon_geography_and_scheduled_skip(case, model, monkeypatch):
    args = args_for(case)
    assert args.steps is None  # Native horizon is read from weights, not duplicated in the CLI.
    target = run(args)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["model"] == "UNet"
    assert manifest["input_steps"] == 12
    assert manifest["output_steps"] == manifest["native_output_steps"] == 6
    assert manifest["masked_output_fraction"] == 0
    norm = Normalization(**manifest["normalization"])
    inputs = (np.log1p(case["rain"][:, ::-1, :]) - norm.mean) / norm.std
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs.copy()).unsqueeze(0))[0, :, 0].numpy()
    expected = norm.inverse(expected)[:, ::-1, :]
    for index, item in enumerate(manifest["outputs"]):
        path = target / item["name"]
        assert sha256(path) == item["sha256"]
        with rasterio.open(path) as dataset:
            assert dataset.crs.to_epsg() == 4326
            assert dataset.xy(0, 0) == pytest.approx((-119.95, 32.95))
            assert dataset.tags()["lead_hours"] == str(index + 1)
            assert dataset.units == ("mm/h",)
            np.testing.assert_allclose(dataset.read(1), expected[index], atol=1e-6)
    assert scheduler.completed_run(target, case["run_time"], "gsmap_now", None, BACKEND)
    other_backend = replace(BACKEND, name="ARAFNO")
    with pytest.raises(ValueError, match="incompatible"):
        scheduler.completed_run(target, case["run_time"], "gsmap_now", 6, other_backend)
    with scheduler.poll_lock(case["outputs"], "gsmap_now", BACKEND) as first:
        with scheduler.poll_lock(case["outputs"], "gsmap_now", other_backend) as second:
            assert first and second
    with pytest.raises(FileExistsError):
        run(args)
    args.steps = 7
    with pytest.raises(ValueError, match="exceeds native horizon"):
        run(args)


def test_missing_hour_and_failed_export_never_publish(case, monkeypatch):
    args = args_for(case)
    first = sorted(case["inputs"].glob("*.gz"))[0]
    absent = first.with_suffix(".hidden")
    first.rename(absent)
    with pytest.raises(ValueError, match="Incomplete hourly window"):
        run(args)
    absent.rename(first)
    original = pipeline.write_geotiff

    def fail_second(path, rain, mask, domain, tags):
        if tags["lead_hours"] == 2:
            raise OSError("simulated disk error")
        original(path, rain, mask, domain, tags)

    monkeypatch.setattr(pipeline, "write_geotiff", fail_second)
    with pytest.raises(OSError, match="disk error"):
        run(args)
    assert not list(case["outputs"].rglob("*.tif"))
    assert not list(case["outputs"].rglob("manifest.json"))
