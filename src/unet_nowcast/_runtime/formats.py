"""Fixed data-format contracts, shared by validation, decoding and export."""

import numpy as np

GSMAP_RESOLUTION = 0.1
GSMAP_WEST, GSMAP_SOUTH, GSMAP_EAST, GSMAP_NORTH = -180.0, -60.0, 180.0, 60.0
GSMAP_DTYPE = np.dtype("<f4")
GSMAP_SHAPE = (
    round((GSMAP_NORTH - GSMAP_SOUTH) / GSMAP_RESOLUTION),
    round((GSMAP_EAST - GSMAP_WEST) / GSMAP_RESOLUTION),
)
GSMAP_BYTES = GSMAP_SHAPE[0] * GSMAP_SHAPE[1] * GSMAP_DTYPE.itemsize
CRS = "EPSG:4326"
PRECIPITATION_UNITS = "mm/h"
