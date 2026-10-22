# UNet — operational GSMaP NOW inference

This independent pipeline consumes 12 consecutive hourly GSMaP NOW observations
and publishes six direct precipitation forecasts as GeoTIFFs in
mm/h. It preserves the reference architecture, weights, and normalization.
Acquisition, preprocessing, export, and scheduling are packaged in
`src/unet_nowcast/_runtime/`; no other model directory or
`../shared/nowcasting_runtime` is required.

## Project layout

```text
04_UNet/
├── config/latam.json           # operational domain, product, and model settings
├── src/unet_nowcast/
│   ├── model.py                # architecture, weights, and inference
│   ├── _runtime/               # private acquisition, processing, GeoTIFF, scheduler
│   └── entrypoint.py           # command-line entrypoint
├── artifacts/v1/               # inference weights, mean, std, SHA-256 manifest
├── reference/                  # supplied training and inference references
├── deploy/                     # systemd timer and installation instructions
├── tools/prepare_artifacts.py  # create a new versioned inference bundle
├── tools/download_artifacts.py # download and verify the public bundle
├── Dockerfile / compose.yaml    # independent CPU container and polling service
├── tests/
└── outputs/                    # generated forecasts, excluded from Git
```

## Download model artifacts

Publish `artifacts/v1/manifest.json` and every file listed in its `files`
section under
`https://ftp.cptec.inpe.br/rainfall/ainpp-pb-latam/checkpoints/unet/v1/`.
After the server files are available over HTTP(S), run from this repository:

```bash
uv run --locked unet-download-artifacts

# Or, without installing dependencies:
python3 tools/download_artifacts.py
```

The command uses the URL recorded in the local manifest, verifies each
download against its pinned SHA-256, skips files already verified, and
atomically installs missing files. It needs only Python's standard library.
Use `--list` to inspect expected URLs without network access, or
`--base-url https://another-host/unet/v1/` to use a mirror with the
same local hashes. The download uses HTTP(S), even though the host is
named `ftp.cptec.inpe.br`.

## Run independently with Docker

The CPU image contains the UNet code, locked runtime dependencies,
configuration, manifest, and normalization files. The checkpoint is not
included. At startup, the container verifies the local bundle and downloads
missing or invalid artifacts from the public HTTP(S) URL in the manifest.
If that server is unavailable, it retries after the configured poll interval;
FTP polling starts only after the bundle is valid.

From `04_UNet`, prepare `.env` with JAXA credentials, your host user and
group IDs, and the poll interval in seconds:

```bash
cp deploy/docker.env.example .env
chmod 600 .env
id -u
id -g
# Edit .env: set FTP_JAXA_USER, FTP_JAXA_PASSWORD, UNET_UID, and UNET_GID.
# Adjust UNET_POLL_INTERVAL_SECONDS if needed; 300 means every five minutes.
mkdir -p outputs
docker compose config --quiet
docker compose up -d --build
docker compose logs -f unet
```

If `.env` already has the FTP credentials, add the three `UNET_*`
settings instead of replacing it. The service polls immediately and then
waits `UNET_POLL_INTERVAL_SECONDS` after each poll. It looks for new
**hourly** GSMaP NOW inputs; polling more frequently reduces detection delay
but does not change the provider's publication cadence. Incomplete windows
and failed transfers are retried on the next poll. Completed issue times are
skipped; a long outage is not automatically backfilled.

`compose.yaml` mounts `artifacts/v1` and `outputs` from the host, so verified
weights and published GeoTIFFs survive container replacement. FTP inputs stay
in the container's temporary directory and are pruned after 24 hours by
scheduled polls. The container runs with your host UID/GID to write the mounted
directories. Use `docker compose down` to stop it. The host needs outbound
HTTPS to the checkpoint server and FTP to JAXA; no port is exposed.

## Install and run

From `04_UNet`, with Python 3.11–3.13:

```bash
uv sync --locked --python 3.12
export FTP_JAXA_USER='your_username'
export FTP_JAXA_PASSWORD='your_password'
uv run --locked unet-nowcast --config config/latam.json \
  --source ftp --output-dir outputs --device cpu
```

The default FTP host is the JAXA host used by the reference. Set
`FTP_JAXA_HOST` to override it. The pipeline selects the latest complete
12-hour window; it does not require future observations. Credentials are read
from the environment and are not written to the forecast manifest.

FTP downloads use a temporary cache at
`<system-temp>/nowcasting-<UID>/unet`. They expire after 24 hours by
default; set `--cache-ttl-hours 6` for a different retention period or
`--cache-dir` for another managed cache. Cleanup runs on every FTP execution
or scheduled poll, including polls with no new forecast. If the timer is
disabled, expired files are removed on the next FTP run or by the operating
system. Files supplied through `--source local --input-dir` are never removed.

To use local GSMaP binaries:

```bash
uv run --locked unet-nowcast --config config/latam.json \
  --source local --input-dir data/raw --run-time 2026-10-21T12:00:00Z \
  --output-dir outputs
```

That example reads observations from 01:00 through 12:00 UTC and publishes
forecasts valid from 13:00 through 18:00 UTC. `--steps 3` publishes only the
first three horizons; these weights do not support more than six. A missing
hour causes an explicit error rather than silently using an older window.

```text
outputs/unet/gsmap_now/YYYY/MM/DD/HHMMZ/
├── unet_<issue>_f001_<valid>.tif
├── ...
├── unet_<issue>_f006_<valid>.tif
└── manifest.json
```

Each GeoTIFF has one float32 band, EPSG:4326, north-up orientation, DEFLATE
compression, UTC timestamps, and precipitation rate in mm/h. The manifest
records inputs, artifacts, hashes, configuration, and software versions.
A complete issue is never overwritten; partial issues are not published.

## Scientific contract and limits

- Finite negative GSMaP values, including −99, become zero mm/h **before**
  `log1p` and z-score normalization. They do not create NoData. NaN and
  infinities follow the explicit missing-data policy in `config/latam.json`.
  Output pixels affected by missing inputs are marked NoData.
- Mean and standard deviation are loaded from the versioned artifact bundle.
  All bundle files are checked against their SHA-256 manifest before inference.
- The reference applies ReLU in normalized space. With the supplied mean,
  the inverse transformation has a minimum near 0.0453 mm/h; this behavior is
  preserved for checkpoint equivalence.
- Reference training and inference used NRT; operational inference uses NOW as
  requested. Meteorological skill with NOW has not yet been established.
- The checkpoint does not verify the training latitude order. The configured
  `north_to_south` orientation is an explicit assumption pending comparison
  with georeferenced training data.
- Training used 320 × 320 patches, whereas this pipeline uses the reference's
  full 880 × 970 LATAM inference grid. Technical compatibility does not prove
  meteorological skill or equivalence to patch-based inference.


## Artifacts, scheduling, and validation

The inference bundle contains tensor weights and is loaded with
`weights_only=True`; the original training checkpoint remains in
`reference/`. To prepare another version, copy the matching statistics and
run `tools/prepare_artifacts.py --checkpoint ... --bundle artifacts/v2`.
The Python wheel contains this model's private runtime, but not its weights or
reference assets. Distribute the wheel, configuration, and versioned bundle
together. Confirm the source license before publishing weights or data.

The lock targets CPU PyTorch. For CUDA, create a separate compatible
environment and install this model's wheel there. The CLI rejects
`--device cuda` if CUDA is unavailable; `auto` selects an available device.

See [deploy/README.md](deploy/README.md) for five-minute systemd polling,
retries, locking, and activation. Installation does not enable the timer.
Run the local checks with:

```bash
uv run --locked pytest
uv run --locked ruff check src tests tools deploy
uv build
```

Recorded validation results and remaining limitations are in
[VALIDATION.md](VALIDATION.md).
