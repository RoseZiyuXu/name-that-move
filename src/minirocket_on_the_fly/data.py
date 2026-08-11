"""Data loading and offline augmentation for motion segments."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tsaug import AddNoise, TimeWarp

# ---------------------------------------------------------------------------
# Default label configuration
# ---------------------------------------------------------------------------

DEFAULT_FILE_NAMES = (
    "all_0_negative",
    "all_1_beginhand",
    "all_2_knee",
    "all_3_retract",
    "all_4_step",
    "all_5_glide",
    "all_6_spillback",
    "all_7_pinwheelbelly",
    "all_8_standturn",
    "all_9_bellycircle",
    "all_10_motif",
)


def load_segments(
    base_path: str | Path,
    file_names: Sequence[str] = DEFAULT_FILE_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all `.pkl` segment files from a structured directory.

    Expected on-disk layout::

        base_path/
            <file_names[0]>/  *.pkl   → label "0"
            <file_names[1]>/  *.pkl   → label "1"
            ...

    Each `.pkl` file must deserialise to a sequence whose first element is an
    ``np.ndarray`` of shape ``(n_channels, n_timesteps)``, e.g. ``(6, 96)``.

    Parameters
    ----------
    base_path:
        Root directory that contains one sub-folder per class.
    file_names:
        Ordered list of sub-folder names.  The list index becomes the integer
        label (stored as ``np.str_``).

    Returns
    -------
    X : ndarray, shape ``(N, n_channels, n_timesteps)``
    y : ndarray of str labels, shape ``(N,)``
    """
    base_path = Path(base_path)
    X: list[np.ndarray] = []
    y: list[np.str_] = []

    for label_idx, folder_name in enumerate(file_names):
        folder_path = base_path / folder_name
        label = np.str_(label_idx)

        if not folder_path.is_dir():
            raise FileNotFoundError(f"Class directory not found: {folder_path}")

        pkl_files = sorted(p for p in folder_path.iterdir() if p.suffix == ".pkl")
        for pkl_path in tqdm(pkl_files, desc=f"Loading '{folder_name}'", unit="file"):
            with pkl_path.open("rb") as fh:
                segment = pickle.load(fh)
            sample = segment if isinstance(segment, np.ndarray) else segment[0]
            sample = np.asarray(sample)
            if sample.ndim != 2:
                raise ValueError(
                    f"Expected a 2D (channels, timesteps) array in {pkl_path}; "
                    f"received shape {sample.shape}"
                )
            X.append(sample)
            y.append(label)

    if not X:
        raise ValueError(f"No .pkl segment files found under {base_path}")

    try:
        X_array = np.stack(X)
    except ValueError as error:
        raise ValueError("All segments must have the same shape") from error

    return X_array, np.array(y)


def augment_segments(
    X: np.ndarray,
    y: np.ndarray,
    n_aug: int = 2,
    noise_scale: float = 0.05,
    max_speed_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Offline augmentation via ``tsaug``.

    Applies ``AddNoise + TimeWarp`` to every sample ``n_aug`` times and
    returns *only* the augmented copies (concatenate with the originals
    yourself if you want both).

    Parameters
    ----------
    X : ndarray, shape ``(N, n_channels, n_timesteps)``
    y : ndarray, shape ``(N,)``
    n_aug:
        Number of augmented copies per sample.
    noise_scale:
        Standard deviation of the Gaussian noise added by ``AddNoise``.
    max_speed_ratio:
        Maximum speed ratio used by ``TimeWarp``.

    Returns
    -------
    X_aug : ndarray, shape ``(N * n_aug, n_channels, n_timesteps)``
    y_aug : ndarray, shape ``(N * n_aug,)``
    """
    # tsaug expects (N, T, C); our storage convention is (N, C, T)
    X_tc = np.transpose(X, (0, 2, 1))  # (N, T, C)

    augmenter = AddNoise(scale=noise_scale) + TimeWarp(max_speed_ratio=max_speed_ratio)

    X_aug: list[np.ndarray] = []
    y_aug: list = []

    for i, x_tc in enumerate(X_tc):
        for _ in range(n_aug):
            x_aug_tc = augmenter.augment(x_tc[np.newaxis, ...])[0]  # (T, C)
            X_aug.append(x_aug_tc.T)  # back to (C, T)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


def make_dataset(
    base_path: str | Path,
    file_names: Sequence[str] = DEFAULT_FILE_NAMES,
    n_aug: int = 2,
    noise_scale: float = 0.05,
    max_speed_ratio: float = 3.0,
    val_fraction: float = 0.2,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """End-to-end helper: load → split → augment the training split.

    Augmentation is intentionally applied only after splitting. This keeps
    transformed copies of a sample out of the validation set and prevents
    train/validation data leakage.

    Returns
    -------
    X : ndarray, shape ``(N_total, n_channels, n_timesteps)``  (float32)
    y : ndarray of str labels, shape ``(N_total,)``
    splits : ``(train_indices, val_indices)`` tuple compatible with ``tsai``
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if n_aug < 0:
        raise ValueError("n_aug must be non-negative")

    X_orig, y_orig = load_segments(base_path, file_names)
    if len(y_orig) < 2:
        raise ValueError("At least two samples are required to create a split")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(y_orig))
    split_point = int(len(y_orig) * (1.0 - val_fraction))
    if split_point == 0 or split_point == len(y_orig):
        raise ValueError("val_fraction produces an empty train or validation split")

    train_orig = indices[:split_point]
    val_indices = indices[split_point:]

    if n_aug:
        # tsaug uses NumPy's random state internally.
        if random_seed is not None:
            np.random.seed(random_seed)
        X_aug, y_aug = augment_segments(
            X_orig[train_orig],
            y_orig[train_orig],
            n_aug=n_aug,
            noise_scale=noise_scale,
            max_speed_ratio=max_speed_ratio,
        )
        X = np.concatenate([X_orig, X_aug], axis=0).astype(np.float32)
        y = np.concatenate([y_orig, y_aug], axis=0)
        augmented_indices = np.arange(len(y_orig), len(y))
        train_indices = np.concatenate([train_orig, augmented_indices])
    else:
        X = X_orig.astype(np.float32)
        y = y_orig
        train_indices = train_orig

    splits = (train_indices, val_indices)

    print(f"Dataset ready — X: {X.shape}, y: {y.shape}")
    return X, y, splits
