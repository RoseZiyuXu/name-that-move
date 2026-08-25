"""Thread-safe latest-value sampling for the real-time workflow."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from threading import Lock
from time import monotonic

import numpy as np

from name_that_move.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    validate_windows,
)


@dataclass(frozen=True)
class WindowDiagnostics:
    """Lightweight quality information collected with one IMU window."""

    started_at: float
    ended_at: float
    osc_message_count: int
    max_channel_age_s: float


@dataclass(frozen=True)
class CompletedWindow:
    """One model-ready IMU window and its acquisition diagnostics."""

    data: np.ndarray
    diagnostics: WindowDiagnostics


class LatestValueWindowBuffer:
    """Sample the latest named OSC values onto a fixed-rate timeline.

    This preserves the strategy used by the working performance prototype:
    incoming channel values update independently, while the sampler takes one
    six-channel snapshot at each configured interval. No interpolation is
    performed.
    """

    def __init__(self, config: IMUWindowConfig = DEFAULT_IMU_CONFIG) -> None:
        """Initialize an empty buffer for the supplied window configuration."""
        self.config = config
        self._channel_indices = {
            name: index for index, name in enumerate(config.channel_names)
        }
        self._values = np.zeros(config.n_channels, dtype=np.float32)
        self._seen = np.zeros(config.n_channels, dtype=bool)
        self._updated_at = np.zeros(config.n_channels, dtype=np.float64)
        self._samples: list[np.ndarray] = []
        self._started_at: float | None = None
        self._message_count = 0
        self._window_message_start = 0
        self._lock = Lock()

    def update(
        self,
        channel_name: str,
        value: float,
        *,
        received_at: float | None = None,
    ) -> None:
        """Update the latest value and arrival time for one named channel."""
        if channel_name not in self._channel_indices:
            raise KeyError(f"Unknown IMU channel: {channel_name}")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("OSC channel values must be numeric")
        if not np.isfinite(value):
            raise ValueError("OSC channel values must be finite")

        timestamp = monotonic() if received_at is None else received_at
        if not np.isfinite(timestamp):
            raise ValueError("received_at must be finite")

        index = self._channel_indices[channel_name]
        with self._lock:
            self._values[index] = value
            self._seen[index] = True
            self._updated_at[index] = timestamp
            self._message_count += 1

    def sample(self, *, sampled_at: float | None = None) -> CompletedWindow | None:
        """Append one latest-value snapshot and return a completed window."""
        timestamp = monotonic() if sampled_at is None else sampled_at
        if not np.isfinite(timestamp):
            raise ValueError("sampled_at must be finite")

        with self._lock:
            if not self._seen.all():
                return None
            if self._started_at is None:
                self._started_at = timestamp
                self._window_message_start = self._message_count

            self._samples.append(self._values.copy())
            if len(self._samples) < self.config.n_timesteps:
                return None

            data = np.asarray(self._samples, dtype=np.float32).T
            diagnostics = WindowDiagnostics(
                started_at=self._started_at,
                ended_at=timestamp,
                osc_message_count=self._message_count - self._window_message_start,
                max_channel_age_s=float(np.max(timestamp - self._updated_at)),
            )
            self._samples.clear()
            self._started_at = None
            self._window_message_start = self._message_count

        normalized = validate_windows(data, config=self.config)[0]
        return CompletedWindow(data=normalized, diagnostics=diagnostics)

