"""Model-specific operations required by the common inference pipeline."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class Backend:
    name: str
    version: str
    load_model: Callable[[Path, torch.device, tuple[int, int]], torch.nn.Module]
    forecast_steps: Callable[[torch.nn.Module, torch.Tensor, int], Iterator[torch.Tensor]]
    default_steps: int | None = None
    library_versions: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.name.lower()
