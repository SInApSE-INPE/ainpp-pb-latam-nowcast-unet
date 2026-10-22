"""Run artifact setup and repeated scheduled FTP polls in one container."""

from __future__ import annotations

import logging
import os
import signal
import threading

from . import download_artifacts

LOG = logging.getLogger(__name__)
POLL_ARGS = [
    "--config",
    "config/latam.json",
    "--source",
    "ftp",
    "--scheduled",
    "--output-dir",
    "outputs",
]


def poll_interval_seconds() -> int:
    raw = os.environ.get("UNET_POLL_INTERVAL_SECONDS", "300")
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError("UNET_POLL_INTERVAL_SECONDS must be a positive integer") from exc
    if interval < 1:
        raise ValueError("UNET_POLL_INTERVAL_SECONDS must be a positive integer")
    return interval


def scheduled_poll() -> int:
    from . import entrypoint

    return entrypoint.main(POLL_ARGS)


def run_loop(stop: threading.Event, interval: int) -> None:
    while not stop.is_set():
        if download_artifacts.main([]) == 0:
            break
        LOG.warning("Model bundle is unavailable; retrying in %s seconds", interval)
        stop.wait(interval)

    while not stop.is_set():
        if scheduled_poll() != 0:
            LOG.warning("Scheduled poll failed; retrying in %s seconds", interval)
        stop.wait(interval)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        interval = poll_interval_seconds()
    except ValueError as exc:
        LOG.error("Container configuration failed: %s", exc)
        return 2

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    LOG.info("Starting UNet with a %s-second poll interval", interval)
    run_loop(stop, interval)
    LOG.info("UNET container stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
