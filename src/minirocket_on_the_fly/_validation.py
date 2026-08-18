"""Internal validation helpers shared by training and inference."""

from __future__ import annotations

from os import PathLike

import numpy as np


def validate_positive_int(value: int, *, name: str) -> int:
    """Return a positive integer or raise a clear configuration error."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_splits(
    splits: tuple[np.ndarray, np.ndarray],
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a non-overlapping ``(train, validation)`` index pair."""
    if not isinstance(splits, (tuple, list)) or len(splits) != 2:
        raise ValueError("splits must contain exactly (train_indices, val_indices)")

    validated = []
    for name, indices in zip(("train_indices", "val_indices"), splits):
        array = np.asarray(indices)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must contain integer indices")
        if np.any(array < 0) or np.any(array >= n_samples):
            raise ValueError(
                f"{name} contains an index outside the valid range "
                f"[0, {n_samples})"
            )
        if np.unique(array).size != array.size:
            raise ValueError(f"{name} must not contain duplicate indices")
        validated.append(array.astype(np.int64, copy=False))

    train_indices, val_indices = validated
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("train_indices and val_indices must not overlap")
    return train_indices, val_indices


def validate_feature_matrix(X_feat: np.ndarray) -> np.ndarray:
    """Validate and normalize MiniRocket features to ``(N, features, 1)``."""
    array = np.asarray(X_feat)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    elif array.ndim != 3 or array.shape[2] != 1:
        raise ValueError(
            "X_feat must have shape (n_samples, n_features) or "
            "(n_samples, n_features, 1)"
        )
    if any(size == 0 for size in array.shape):
        raise ValueError("X_feat must not contain empty dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("X_feat must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError("X_feat must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_labels(y: np.ndarray, *, n_samples: int) -> np.ndarray:
    """Validate a one-dimensional label array aligned with the inputs."""
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError("y must have shape (n_samples,)")
    if array.size != n_samples:
        raise ValueError(
            f"X and y must contain the same number of samples; received "
            f"{n_samples} and {array.size}"
        )
    return array


def validate_tag(tag: str) -> str:
    """Validate a non-empty filename-safe model tag."""
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not tag.strip():
        raise ValueError("tag must not be empty")
    if any(separator in tag for separator in ("/", "\\")) or tag in {".", ".."}:
        raise ValueError("tag must not contain path separators")
    return tag


def validate_path(path: str | PathLike[str], *, name: str) -> str | PathLike[str]:
    """Reject path values that cannot be interpreted by ``pathlib.Path``."""
    if not isinstance(path, (str, PathLike)):
        raise TypeError(f"{name} must be a path-like value")
    return path
