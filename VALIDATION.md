# UNet validation record

Recorded October 21, 2026.

## Checkpoint and technical checks

The supplied checkpoint loaded strictly into the reference architecture.
The versioned inference bundle contains only tensor weights and matching
mean/std files, with SHA-256 values in its manifest. Normalization files
match the originals referenced by the source script.

Six initial tests passed, covering exact reference equivalence on a
64 × 64 deterministic input, 4D/5D input support, model output leads,
rejection of incompatible weights or horizons, georeferenced GeoTIFFs,
atomic publication, and idempotent scheduling. Ruff passed. The service
and timer were rendered and passed `systemd-analyze verify`; neither was
installed or activated. The six leads are predicted in one model call.

A full-grid run with **12 synthetic GSMaP binaries** and the real checkpoint
produced six 880 × 970, EPSG:4326, float32 GeoTIFFs in mm/h. Reopened files
had the expected UTC leads and finite nonnegative values. The run included
a −99 input value, clamped to zero before normalization. Temporary synthetic
inputs were removed. These results demonstrate software behavior, **not
meteorological forecast quality**.

## Environment and open validation

The earlier tests and full-grid run used the existing CPU environment in `02_Xception/.venv`, with PyTorch 2.10.0 and rasterio 1.5.2.

A clean installation from the lock was not completed in the initial offline
session because the required CPU PyTorch wheel was unavailable or online
dependency retrieval failed through DNS. Run `uv sync --locked` with access
to the configured package indexes before operational use. The lock itself
passed `uv lock --offline --check`.

Real JAXA FTP, GPU inference, and the active systemd service were not tested
at this stage. Training latitude order remains unverified, and the NRT-to-NOW
input change requires meteorological evaluation.

## Independent distribution

Acquisition, processing, publication, and scheduling are embedded in
`src/unet_nowcast/_runtime/` and packaged in this model's wheel.
`uv lock --offline` and `uv build --offline` passed before the retention
change. The inspected wheel contains all 11 private runtime modules and
has no dependency on `nowcasting-runtime` or `../shared`. Importing the
runtime from an extracted wheel outside the checkout passed. The six
initial tests passed again. Changes needed by multiple independent models
must be applied to each one.

## Temporary download retention

FTP inputs now default to the system temporary directory, isolated by user
and model. The default 24-hour retention is configurable with
`--cache-ttl-hours`. Cleanup removes expired managed downloads on each FTP
run or scheduled poll, including polls with no new forecast, and preserves
local inputs.

**Eight tests passed** after this change; Ruff and
`uv lock --offline --check` passed. Tests used the project's available environment. The current offline
wheel rebuild could not run because `hatchling` was missing from the build
cache; the previous build had passed.

## Docker deployment — October 22, 2026

The independent CPU image, Compose service, startup checkpoint verification, retry loop, polling interval, and bind mounts were added. Twenty tests passed, including container retry and configuration tests. Ruff checks, formatting, lock consistency, and static Compose/context validation passed. The image could not be built here: the Docker command is backed by Podman, which cannot initialize its state on the read-only `/run/user` filesystem. A live Docker build, checkpoint HTTP fetch, and JAXA FTP poll remain to be verified on the deployment host.
