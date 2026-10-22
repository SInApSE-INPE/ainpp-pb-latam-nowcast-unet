"""The public artifact command must install only files matching the local manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "unet_nowcast" / "download_artifacts.py"
SPEC = importlib.util.spec_from_file_location("artifact_downloader", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class ArtifactDownloadTests(unittest.TestCase):
    def test_downloads_verified_bundle_and_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            content = b"small test checkpoint"
            expected = hashlib.sha256(content).hexdigest()
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model_file": "model.pt",
                        "sha256": expected,
                        "download_base_url": "https://example.test/checkpoints/irene/v1/",
                    }
                )
            )
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)) as fetch:
                self.assertEqual(downloader.main(["--bundle", str(bundle), "--attempts", "1"]), 0)
                self.assertEqual(
                    fetch.call_args.args[0],
                    "https://example.test/checkpoints/irene/v1/model.pt",
                )
            self.assertEqual((bundle / "model.pt").read_bytes(), content)
            with patch.object(downloader, "urlopen", side_effect=AssertionError("fetched again")):
                self.assertEqual(downloader.main(["--bundle", str(bundle)]), 0)
            (bundle / "model.pt").write_bytes(b"corrupt")
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(content)):
                self.assertEqual(downloader.main(["--bundle", str(bundle)]), 0)
            self.assertEqual((bundle / "model.pt").read_bytes(), content)

    def test_downloads_every_gsmap_bundle_file(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            contents = {
                "checkpoint.pth": b"weights",
                "mean+log.npy": b"mean",
                "std.npy": b"std",
            }
            roles = {"checkpoint": "checkpoint.pth", "mean": "mean+log.npy", "std": "std.npy"}
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": {
                            role: {
                                "name": name,
                                "sha256": hashlib.sha256(contents[name]).hexdigest(),
                            }
                            for role, name in roles.items()
                        },
                        "download_base_url": "https://example.test/model/v1/",
                    }
                )
            )

            def fetch(url, timeout):
                name = url.rsplit("/", 1)[-1].replace("%2B", "+")
                return io.BytesIO(contents[name])

            with patch.object(downloader, "urlopen", side_effect=fetch) as request:
                self.assertEqual(downloader.main(["--bundle", str(bundle)]), 0)
                self.assertEqual(request.call_count, 3)
            for name, content in contents.items():
                self.assertEqual((bundle / name).read_bytes(), content)

    def test_rejects_bad_hash_without_publishing_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": {
                            role: {"name": f"{role}.npy", "sha256": "0" * 64}
                            for role in ("checkpoint", "mean", "std")
                        },
                        "download_base_url": "https://example.test/model/v1/",
                    }
                )
            )
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"wrong")):
                self.assertEqual(downloader.main(["--bundle", str(bundle), "--attempts", "1"]), 1)
            self.assertFalse((bundle / "checkpoint.npy").exists())
            self.assertEqual(list(bundle.glob("*.part")), [])
            self.assertEqual(list(bundle.glob(".*.part")), [])

    def test_retries_transient_download_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            content = b"recovered checkpoint"
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model_file": "model.pt",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "download_base_url": "https://example.test/model/v1/",
                    }
                )
            )
            with (
                patch.object(
                    downloader,
                    "urlopen",
                    side_effect=[OSError("temporarily unavailable"), io.BytesIO(content)],
                ) as fetch,
                patch.object(downloader.time, "sleep") as pause,
            ):
                self.assertEqual(downloader.main(["--bundle", str(bundle), "--attempts", "2"]), 0)
                self.assertEqual(fetch.call_count, 2)
                pause.assert_called_once()
            self.assertEqual((bundle / "model.pt").read_bytes(), content)

    def test_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model_file": "../outside.pt",
                        "sha256": "0" * 64,
                        "download_base_url": "https://example.test/model/v1/",
                    }
                )
            )
            with patch.object(
                downloader, "urlopen", side_effect=AssertionError("should not fetch")
            ):
                self.assertEqual(downloader.main(["--bundle", str(bundle)]), 1)
            self.assertFalse((bundle.parent / "outside.pt").exists())


if __name__ == "__main__":
    unittest.main()
