"""Load saved IMU windows without importing model-training libraries."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from name_that_move._validation import validate_path
from name_that_move.preprocessing import validate_windows


def load_segment(path: str | Path) -> np.ndarray:
    """Load one channel-first IMU window from a ``.pkl`` file.

    The file may contain either a bare array of shape
    ``(n_channels, n_timesteps)`` or a sequence whose first item is one.

    Parameters
    ----------
    path:
        Path to the saved window.

    Returns
    -------
    ndarray, shape ``(n_channels, n_timesteps)``
        Loaded numeric window in channel-first format.

    """
    path = Path(validate_path(path, name="path"))
    with path.open("rb") as file_handle:
        data = pickle.load(file_handle)

    if not isinstance(data, np.ndarray):
        try:
            data = data[0]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                "segment file must contain an array or a sequence whose first "
                "item is an array"
            ) from error

    batch = validate_windows(data)
    if len(batch) != 1:
        raise ValueError("segment file must contain exactly one window")
    return batch[0]


def load_segments_batch(path: str | Path) -> np.ndarray:
    """Load a channel-first batch of IMU windows from a ``.pkl`` file.

    Parameters
    ----------
    path:
        Path to a file containing an existing window batch.

    Returns
    -------
    ndarray, shape ``(N, n_channels, n_timesteps)``
        Loaded windows as a contiguous ``float32`` batch.

    """
    path = Path(validate_path(path, name="path"))
    with path.open("rb") as file_handle:
        data = pickle.load(file_handle)

    return validate_windows(data)
