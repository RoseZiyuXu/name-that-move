"""Data loading and augmentation for the offline training workflow.

An IMU window is one fixed-duration segment of continuous sensor data with
channel-first shape ``(n_channels, n_timesteps)``.
"""

from __future__ import annotations

import pickle
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tsaug import AddNoise, TimeWarp

from name_that_move.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    validate_windows,
)

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
    *,
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
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
    config:
        Recording/model configuration every segment must match. The default
        is 48 Hz, two seconds, and the standard six-channel order.

    Returns
    -------
    X : ndarray, shape ``(N, n_channels, n_timesteps)``
        Loaded IMU windows in channel-first format.
    y : ndarray of str labels, shape ``(N,)``
        Class labels derived from the folder order.

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
            X.append(_load_window(pkl_path, config=config))
            y.append(label)

    if not X:
        raise ValueError(f"No .pkl segment files found under {base_path}")

    try:
        X_array = np.stack(X)
    except ValueError as error:
        raise ValueError("All segments must have the same shape") from error

    return X_array, np.array(y)


def make_session_dataset(
    base_path: str | Path,
    class_folders: Mapping[str, Sequence[str]],
    validation_folders: Collection[str],
    *,
    n_aug: int = 0,
    noise_scale: float = 0.05,
    max_speed_ratio: float = 3.0,
    random_seed: int | None = None,
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Load class/session folders and hold out complete recording sessions.

    Unlike :func:`make_dataset`, this helper does not randomly divide windows.
    The caller explicitly identifies complete folders to use for validation,
    preventing highly related windows from one recording session from appearing
    in both training and validation. Here, a recording session means a separate
    acquisition period or recording round. Holding out whole sessions measures
    whether the model transfers to a later recording instead of benefiting from
    session-specific similarity between neighboring windows.

    Parameters
    ----------
    base_path:
        Root directory containing the session folders.
    class_folders:
        Mapping from public class label to its recording-session folders. For
        example, ``{"still": ("still1", "still2", "still3")}`` groups all
        three folders under the label ``"still"``.
    validation_folders:
        Session-folder names reserved entirely for validation. Every class must
        contribute at least one training folder and one validation folder.
    n_aug:
        Number of augmented copies generated per training window. Validation
        windows are never augmented.
    noise_scale:
        Standard deviation used by the noise augmenter.
    max_speed_ratio:
        Maximum speed ratio used by the time-warp augmenter.
    random_seed:
        Optional NumPy seed for reproducible augmentation.
    config:
        Recording/model configuration every window must match.

    Returns
    -------
    X : ndarray, shape ``(N_total, n_channels, n_timesteps)``
        Original windows followed by any augmented training windows.
    y : ndarray of str labels, shape ``(N_total,)``
        Public class labels supplied through ``class_folders``.
    splits : tuple of ndarray
        Non-overlapping training and validation indices compatible with tsai.

    """
    base_path = Path(base_path)
    if not isinstance(class_folders, Mapping) or not class_folders:
        raise ValueError("class_folders must be a non-empty mapping")
    if n_aug < 0:
        raise ValueError("n_aug must be non-negative")
    if isinstance(validation_folders, (str, bytes)):
        raise TypeError("validation_folders must be a collection of folder names")

    validation_set = set(validation_folders)
    if not validation_set:
        raise ValueError("validation_folders must not be empty")
    if any(not isinstance(folder, str) or not folder for folder in validation_set):
        raise TypeError("validation_folders must contain non-empty strings")

    normalized: list[tuple[str, tuple[str, ...]]] = []
    folder_to_label: dict[str, str] = {}
    for label, folders in class_folders.items():
        if not isinstance(label, str) or not label:
            raise TypeError("class_folders labels must be non-empty strings")
        if isinstance(folders, (str, bytes)):
            raise TypeError(f"Folders for class '{label}' must be a sequence")
        folder_tuple = tuple(folders)
        if not folder_tuple:
            raise ValueError(f"Class '{label}' must contain at least one folder")
        for folder in folder_tuple:
            if not isinstance(folder, str) or not folder:
                raise TypeError("class folder names must be non-empty strings")
            if folder in folder_to_label:
                raise ValueError(
                    f"Session folder '{folder}' is assigned to both "
                    f"'{folder_to_label[folder]}' and '{label}'"
                )
            folder_to_label[folder] = label
        normalized.append((label, folder_tuple))

    unknown_validation = validation_set.difference(folder_to_label)
    if unknown_validation:
        unknown = ", ".join(sorted(unknown_validation))
        raise ValueError(f"Unknown validation folder(s): {unknown}")

    for label, folders in normalized:
        class_validation = validation_set.intersection(folders)
        class_training = set(folders).difference(validation_set)
        if not class_training or not class_validation:
            raise ValueError(
                f"Class '{label}' must have at least one training folder and "
                "one validation folder"
            )

    X_orig: list[np.ndarray] = []
    y_orig: list[str] = []
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label, folders in normalized:
        for folder in folders:
            folder_path = base_path / folder
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Session directory not found: {folder_path}")
            pkl_files = sorted(path for path in folder_path.iterdir() if path.suffix == ".pkl")
            if not pkl_files:
                raise ValueError(f"No .pkl windows found in session: {folder_path}")
            split_indices = val_indices if folder in validation_set else train_indices
            for pkl_path in tqdm(pkl_files, desc=f"Loading '{folder}'", unit="file"):
                split_indices.append(len(X_orig))
                X_orig.append(_load_window(pkl_path, config=config))
                y_orig.append(label)

    X = np.stack(X_orig).astype(np.float32)
    y = np.asarray(y_orig)
    train_array = np.asarray(train_indices)
    val_array = np.asarray(val_indices)

    if n_aug:
        if random_seed is not None:
            np.random.seed(random_seed)
        X_aug, y_aug = augment_segments(
            X[train_array],
            y[train_array],
            n_aug=n_aug,
            noise_scale=noise_scale,
            max_speed_ratio=max_speed_ratio,
        )
        first_augmented = len(X)
        X = np.concatenate([X, X_aug], axis=0).astype(np.float32)
        y = np.concatenate([y, y_aug], axis=0)
        train_array = np.concatenate(
            [train_array, np.arange(first_augmented, len(X))]
        )

    print(
        f"Session dataset ready — X: {X.shape}, train: {len(train_array)}, "
        f"validation: {len(val_array)}"
    )
    return X, y, (train_array, val_array)


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
        Original IMU windows to augment.
    y : ndarray, shape ``(N,)``
        Labels aligned with the windows in ``X``.
    n_aug:
        Number of augmented copies per sample.
    noise_scale:
        Standard deviation of the Gaussian noise added by ``AddNoise``.
    max_speed_ratio:
        Maximum speed ratio used by ``TimeWarp``.

    Returns
    -------
    X_aug : ndarray, shape ``(N * n_aug, n_channels, n_timesteps)``
        Generated augmented windows.
    y_aug : ndarray, shape ``(N * n_aug,)``
        Labels repeated for each generated window.

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
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """End-to-end helper: load → split → augment the training split.

    Augmentation is intentionally applied only after splitting. This keeps
    transformed copies of a sample out of the validation set and prevents
    train/validation data leakage.

    Parameters
    ----------
    base_path : str or Path
        Root directory that contains one sub-folder per class.
    file_names : sequence of str
        Ordered class-folder names passed to ``load_segments``.
    n_aug : int
        Number of augmented copies generated per training sample.
    noise_scale : float
        Standard deviation used by the noise augmenter.
    max_speed_ratio : float
        Maximum speed ratio used by the time-warp augmenter.
    val_fraction : float
        Fraction of original samples reserved for validation.
    random_seed : int or None
        Optional seed for reproducible splitting and augmentation.
    config : IMUWindowConfig
        Configuration that every loaded window must match.

    Returns
    -------
    X : ndarray, shape ``(N_total, n_channels, n_timesteps)``  (float32)
        Original and augmented model inputs.
    y : ndarray of str labels, shape ``(N_total,)``
        Labels aligned with ``X``.
    splits : ``(train_indices, val_indices)`` tuple compatible with ``tsai``
        Non-overlapping indices for training and validation.

    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if n_aug < 0:
        raise ValueError("n_aug must be non-negative")

    X_orig, y_orig = load_segments(base_path, file_names, config=config)
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


def _load_window(pkl_path: Path, *, config: IMUWindowConfig) -> np.ndarray:
    """Load one pickled IMU window and report its path on validation errors."""
    with pkl_path.open("rb") as file:
        segment = pickle.load(file)
    sample = segment if isinstance(segment, np.ndarray) else segment[0]
    try:
        return validate_windows(sample, config=config)[0]
    except TypeError as error:
        raise TypeError(f"Invalid IMU window in {pkl_path}: {error}") from error
    except ValueError as error:
        raise ValueError(f"Invalid IMU window in {pkl_path}: {error}") from error
