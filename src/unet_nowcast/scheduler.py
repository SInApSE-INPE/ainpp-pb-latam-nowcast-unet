"""U-Net adapter for scheduled polling."""

from ._runtime import scheduler
from .entrypoint import BACKEND


def poll_once(args):
    return scheduler.poll_once(args, BACKEND)
