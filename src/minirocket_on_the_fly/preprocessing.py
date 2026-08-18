"""Preprocess continuous IMU samples into model-ready windows.

An IMU window is one fixed-duration segment of a continuous sensor stream,
stored in channel-first shape ``(n_channels, n_timesteps)``. For example, the
default configuration produces a two-second, six-channel window sampled at
48 Hz with shape ``(6, 96)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CHANNEL_NAMES = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


@dataclass(frozen=True)
class IMUWindowConfig:
    """Configuration shared by IMU recording, training, and inference.

    The default matches the current performance pipeline: one six-axis IMU,
    sampled at 48 Hz in non-overlapping two-second windows.
    """

    sample_rate_hz: float = 48.0
    window_duration_s: float = 2.0
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES

    def __post_init__(self) -> None:
        """Validate the recording configuration after initialization."""
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")
        if not self.channel_names:
            raise ValueError("channel_names must not be empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")

        exact_size = self.sample_rate_hz * self.window_duration_s
        rounded_size = round(exact_size)
        if not np.isclose(exact_size, rounded_size):
            raise ValueError(
                "sample_rate_hz * window_duration_s must produce a whole "
                "number of samples"
            )

    @property
    def n_channels(self) -> int:
        """Number of channels expected in each sample."""
        return len(self.channel_names)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps expected in each window."""
        return round(self.sample_rate_hz * self.window_duration_s)


DEFAULT_IMU_CONFIG = IMUWindowConfig()


def validate_windows(
    windows: np.ndarray,
    *,
    config: IMUWindowConfig | None = None,
) -> np.ndarray:
    """Validate and normalize one window or a batch of windows.

    Parameters
    ----------
    windows:
        A single ``(channels, timesteps)`` window or a batch with shape
        ``(n_windows, channels, timesteps)``.
    config:
        When provided, require the channel and timestep dimensions to match
        this recording/model configuration. When omitted, only the rank,
        non-empty dimensions, and finite values are validated.

    Returns
    -------
    ndarray
        A contiguous ``float32`` batch with shape
        ``(n_windows, channels, timesteps)``.

    """
    array = np.asarray(windows)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(
            "windows must have shape (channels, timesteps) or "
            "(n_windows, channels, timesteps)"
        )

    if any(size == 0 for size in array.shape):
        raise ValueError("windows must not contain empty dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("windows must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("windows must contain only finite values")

    if config is not None:
        expected = (config.n_channels, config.n_timesteps)
        received = tuple(array.shape[1:])
        if received != expected:
            raise ValueError(
                f"Expected window shape {expected} from the model/data "
                f"configuration; received {received}"
            )

    return np.ascontiguousarray(array, dtype=np.float32)


def make_windows(
    samples: np.ndarray,
    *,
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
    stride: int | None = None,
) -> np.ndarray:
    """Split a continuous IMU stream into model-ready windows.

    ``samples`` uses the streaming-friendly shape ``(timesteps, channels)``.
    Incomplete samples at the end are dropped. By default, ``stride`` equals
    the configured window size, producing non-overlapping windows.

    Parameters
    ----------
    samples : ndarray
        Continuous numeric samples with shape ``(timesteps, channels)``.
    config : IMUWindowConfig
        Channel order and window-size configuration.
    stride : int or None
        Number of timesteps between window starts. ``None`` creates
        non-overlapping windows.

    Returns
    -------
    ndarray
        A contiguous ``float32`` batch with shape
        ``(n_windows, channels, timesteps)``.

    """
    array = np.asarray(samples)
    if array.ndim != 2:
        raise ValueError("samples must have shape (timesteps, channels)")
    if array.shape[1] != config.n_channels:
        raise ValueError(
            f"Expected {config.n_channels} channels in the order "
            f"{config.channel_names}; received {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("samples must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("samples must contain only finite values")

    window_size = config.n_timesteps
    stride = window_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(array) < window_size:
        raise ValueError(
            f"At least {window_size} timesteps are required; received {len(array)}"
        )

    starts = range(0, len(array) - window_size + 1, stride)
    windows = np.stack([array[start : start + window_size].T for start in starts])
    return validate_windows(windows, config=config)
