from pathlib import Path

from unet_nowcast._runtime.deployment import render_units

if __name__ == "__main__":
    render_units(Path(__file__).resolve().parent.parent, "unet_nowcast", "UNet", "unet-now")
