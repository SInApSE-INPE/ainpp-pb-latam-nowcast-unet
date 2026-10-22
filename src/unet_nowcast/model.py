"""U-Net architecture and direct multi-hour inference matching the supplied checkpoint."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HIDDEN_CHANNELS = (64, 128, 256, 512, 1024)
KERNEL_SIZE = 7


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, mid_channels: int | None = None
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(
            in_channels + out_channels, out_channels, kernel_size, in_channels // 2
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y, diff_x = x2.shape[-2] - x1.shape[-2], x2.shape[-1] - x1.shape[-1]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: tuple[int, ...] = HIDDEN_CHANNELS,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__()
        if len(hidden_channels) < 2 or min(hidden_channels) < 1 or kernel_size < 1:
            raise ValueError("Invalid U-Net architecture")
        effective = (*hidden_channels[:-1], hidden_channels[-1] // 2)
        self.inc = DoubleConv(in_channels, effective[0], kernel_size)
        self.down_blocks = nn.ModuleList(
            Down(effective[i], effective[i + 1], kernel_size) for i in range(len(effective) - 1)
        )
        self.up_blocks = nn.ModuleList(
            Up(effective[i], effective[i - 1], kernel_size)
            for i in range(len(effective) - 1, 0, -1)
        )
        self.outc = nn.Conv2d(effective[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.down_blocks:
            skips.append(down(skips[-1]))
        x = skips[-1]
        for up, skip in zip(self.up_blocks, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        return self.outc(x)


class PrecipitationForecaster(nn.Module):
    def __init__(self, input_timesteps: int, output_timesteps: int, img_size: tuple[int, int]):
        super().__init__()
        if min(input_timesteps, output_timesteps) < 1 or min(img_size) < 16:
            raise ValueError("Positive sequence lengths and image dimensions >= 16 are required")
        self.input_steps = input_timesteps
        self.output_steps = output_timesteps
        self.img_size = tuple(img_size)
        self.unet = UNet(input_timesteps, output_timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            if x.shape[2] != 1:
                raise ValueError("Expected one precipitation channel per timestep")
            x = x.squeeze(2)
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.input_steps, *self.img_size):
            raise ValueError(f"Expected [batch, {self.input_steps}, {self.img_size}]")
        prediction = self.unet(x)
        if prediction.shape[-2:] != self.img_size:
            raise ValueError("U-Net output dimensions differ from the configured domain")
        # Match the original: ReLU is applied before inverse log-zscore.
        return torch.relu(prediction).unsqueeze(2)


def load_model(path: Path, device: torch.device, expected_shape: tuple[int, int]):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Empty or invalid state dictionary")
    while all(k.startswith("module.") for k in state) or all(
        k.startswith("_orig_mod.") for k in state
    ):
        state = {k.split(".", 1)[1]: v for k, v in state.items()}
    try:
        stem = state["unet.inc.double_conv.0.weight"]
        head = state["unet.outc.weight"]
        input_steps, output_steps = stem.shape[1], head.shape[0]
        if tuple(stem.shape[-2:]) != (KERNEL_SIZE, KERNEL_SIZE):
            raise ValueError("Unsupported U-Net kernel size")
    except (KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Not a supported U-Net checkpoint") from exc
    model = PrecipitationForecaster(input_steps, output_steps, expected_shape)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def forecast_steps(model: PrecipitationForecaster, inputs: torch.Tensor, steps: int):
    """Run the network once and yield requested forecast hours chronologically."""
    if not 1 <= steps <= model.output_steps:
        raise ValueError(f"steps must be between 1 and native horizon {model.output_steps}")
    model.eval()
    with torch.inference_mode():
        prediction = model(inputs)
        if prediction.shape != (inputs.shape[0], model.output_steps, 1, *model.img_size):
            raise ValueError("Invalid U-Net output shape")
        if not torch.isfinite(prediction).all():
            raise ValueError("Model produced non-finite predictions")
    for lead in range(steps):
        yield prediction[:, lead]
