import os
import tempfile
from pathlib import Path

import pytest

from unet_nowcast._runtime.acquisition import prune_cache
from unet_nowcast._runtime.pipeline import validate_runtime
from unet_nowcast.entrypoint import parser


def test_ftp_cache_defaults_to_temporary_storage_with_positive_ttl():
    args = parser().parse_args(["--config", "unused.json"])
    assert args.cache_dir.is_relative_to(Path(tempfile.gettempdir()))
    assert args.cache_ttl_hours == 24
    args.cache_ttl_hours = 0
    with pytest.raises(ValueError, match="cache-ttl-hours"):
        validate_runtime(args)


def test_prune_removes_expired_downloads_only(tmp_path):
    cache = tmp_path / "cache"
    managed = cache / "0123456789abcdef"
    managed.mkdir(parents=True)
    foreign = cache / "user-inputs"
    foreign.mkdir()
    old = managed / "gsmap_now.20230101.0000.dat.gz"
    fresh = managed / "gsmap_now.20230102.0000.dat.gz"
    partial = managed / "interrupted.part"
    unrelated = foreign / "gsmap_now.20230101.0000.dat.gz"
    outside = tmp_path / "local-input.dat.gz"
    for path in (old, fresh, partial, unrelated, outside):
        path.write_bytes(b"input")
    now = 200_000.0
    for path in (old, partial, unrelated, outside):
        os.utime(path, (now - 25 * 3600, now - 25 * 3600))
    os.utime(fresh, (now - 23 * 3600, now - 23 * 3600))
    link = managed / "linked.dat.gz"
    link.symlink_to(outside)

    assert prune_cache(cache, 24, now=now) == 2
    assert not old.exists() and not partial.exists()
    assert fresh.exists() and unrelated.exists() and outside.exists() and link.is_symlink()
    assert prune_cache(cache, 24, now=now) == 0
