"""Container startup retries and scheduled polling use one configured cadence."""

import pytest

from unet_nowcast import container


class StopAfterWaits:
    def __init__(self, count):
        self.count = count
        self.waits = []

    def is_set(self):
        return len(self.waits) >= self.count

    def wait(self, seconds):
        self.waits.append(seconds)


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5"])
def test_poll_interval_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("UNET_POLL_INTERVAL_SECONDS", value)
    with pytest.raises(ValueError, match="positive integer"):
        container.poll_interval_seconds()


def test_poll_interval_uses_configured_seconds(monkeypatch):
    monkeypatch.setenv("UNET_POLL_INTERVAL_SECONDS", "420")
    assert container.poll_interval_seconds() == 420


def test_checkpoint_retry_precedes_first_poll(monkeypatch):
    order = []
    results = iter([1, 0])

    def download(args):
        order.append(("download", args))
        return next(results)

    def poll():
        order.append(("poll", container.POLL_ARGS))
        return 0

    monkeypatch.setattr(container.download_artifacts, "main", download)
    monkeypatch.setattr(container, "scheduled_poll", poll)
    stop = StopAfterWaits(2)
    container.run_loop(stop, 300)
    assert [event for event, _ in order] == ["download", "download", "poll"]
    assert order[-1][1] == container.POLL_ARGS
    assert stop.waits == [300, 300]


def test_failed_poll_retries_on_next_interval(monkeypatch):
    polls = []
    monkeypatch.setattr(container.download_artifacts, "main", lambda args: 0)

    def poll():
        polls.append(container.POLL_ARGS)
        return 1 if len(polls) == 1 else 0

    monkeypatch.setattr(container, "scheduled_poll", poll)
    stop = StopAfterWaits(2)
    container.run_loop(stop, 60)
    assert len(polls) == 2
    assert stop.waits == [60, 60]
