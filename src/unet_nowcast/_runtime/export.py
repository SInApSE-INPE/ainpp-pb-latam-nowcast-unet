"""North-up, pixel-area GeoTIFF output with explicit time and provenance metadata."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .config import Domain
from .formats import CRS, PRECIPITATION_UNITS

NODATA = -9999.0


def write_geotiff(path: Path, rain: np.ndarray, mask: np.ndarray, domain: Domain, tags: dict):
    if rain.shape != domain.shape or mask.shape != domain.shape:
        raise ValueError("Output dimensions differ from configured domain")
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Output precipitation must be finite and nonnegative")
    data = np.where(mask, NODATA, rain).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=domain.shape[0],
        width=domain.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(domain.west, domain.north, domain.resolution, domain.resolution),
        nodata=NODATA,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    ) as dataset:
        dataset.write(data, 1)
        dataset.set_band_description(1, "precipitation_rate")
        dataset.set_band_unit(1, PRECIPITATION_UNITS)
        dataset.update_tags(AREA_OR_POINT="Area", units=PRECIPITATION_UNITS, **tags)
