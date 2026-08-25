"""MiniRocket feature extraction and offline linear-head training.

An IMU window is one fixed-duration segment of continuous sensor data with
channel-first shape ``(n_channels, n_timesteps)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tsai.basics import (
    Learner,
    ShowGraph,
    TSClassification,
    TSStandardize,
    accuracy,
    get_ts_dls,
    timer,
)
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures, MiniRocketHead
from tsai.models.utils import build_ts_model

from name_that_move._compat import default_device, get_minirocket_features
from name_that_move._validation import (
    validate_feature_matrix,
    validate_labels,
    validate_path,
    validate_positive_int,
    validate_splits,
    validate_tag,
)
from name_that_move.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    validate_windows,
)


def extract_features(
    X: np.ndarray,
    splits: tuple[np.ndarray, np.ndarray],
    chunksize: int = 32,
    *,
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
) -> tuple[np.ndarray, MiniRocketFeatures]:
    """Fit a ``MiniRocketFeatures`` model and extract features for all samples.

    Parameters
    ----------
    X:
        Float32 array of shape ``(N, n_channels, n_timesteps)``.
    splits:
        ``(train_indices, val_indices)`` tuple.  Only the training split is
        used to fit the feature extractor.
    chunksize:
        Batch size used during feature extraction (lower = less GPU memory).
    config:
        Configuration that all input windows must match.

    Returns
    -------
    X_feat : ndarray, shape ``(N, n_features, 1)``
        Extracted MiniRocket features for every input window.
    mrf    : fitted ``MiniRocketFeatures`` instance (on ``default_device()``)
        Fitted feature extractor used to transform the windows.

    """
    X = validate_windows(X, config=config)
    chunksize = validate_positive_int(chunksize, name="chunksize")
    train_indices, _ = validate_splits(splits, n_samples=len(X))
    X_train = X[train_indices]

    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device()).float()
    mrf._minirocket_input_shape = tuple(X.shape[1:])
    mrf.fit(X_train)

    X_feat = get_minirocket_features(X, mrf, chunksize=chunksize, to_np=True)
    print(f"Features extracted — shape: {X_feat.shape}")
    return X_feat, mrf


def train(
    X_feat: np.ndarray,
    y: np.ndarray,
    splits: tuple[np.ndarray, np.ndarray],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float | None = None,
    show_graph: bool = False,
) -> Learner:
    """Train a ``MiniRocketHead`` linear classifier on pre-extracted features.

    If ``lr`` is ``None`` the learning-rate finder is run automatically and
    its ``valley`` suggestion is used.

    Parameters
    ----------
    X_feat : ndarray, shape ``(N, n_features, 1)``
        Pre-extracted MiniRocket feature matrix.
    y : ndarray of str labels, shape ``(N,)``
        Class label for each row of ``X_feat``.
    splits : tuple of ndarray
        Non-overlapping ``(train_indices, val_indices)`` arrays.
    epochs : int
        Number of ``fit_one_cycle`` epochs.
    batch_size : int
        DataLoader batch size.
    lr : float or None
        Fixed learning rate, or ``None`` to run ``lr_find``.
    show_graph : bool
        Display an updating training graph. Keep disabled in terminals and CI;
        enable explicitly in a compatible notebook environment.

    Returns
    -------
    learn : trained ``fastai`` ``Learner``
        Learner containing the fitted classification head.

    """
    X_feat = validate_feature_matrix(X_feat)
    y = validate_labels(y, n_samples=len(X_feat))
    splits = validate_splits(splits, n_samples=len(X_feat))
    epochs = validate_positive_int(epochs, name="epochs")
    batch_size = validate_positive_int(batch_size, name="batch_size")
    if not isinstance(show_graph, bool):
        raise TypeError("show_graph must be a boolean")
    if lr is not None:
        if isinstance(lr, bool) or not isinstance(lr, (int, float, np.number)):
            raise TypeError("lr must be a positive finite number or None")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be a positive finite number or None")
        lr = float(lr)

    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)

    if lr is None:
        # Use a small batch size for the LR finder to keep it fast
        dls_find = get_ts_dls(
            X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=16
        )
        model_find = build_ts_model(MiniRocketHead, dls=dls_find)
        learn_find = Learner(dls_find, model_find, metrics=accuracy)
        lr_result = learn_find.lr_find()
        lr = float(lr_result.valley)
        print(f"LR finder suggested: {lr:.2e}")

    dls = get_ts_dls(
        X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=batch_size
    )
    model = build_ts_model(MiniRocketHead, dls=dls)
    callbacks = [ShowGraph()] if show_graph else None
    learn = Learner(dls, model, metrics=accuracy, cbs=callbacks)

    timer.start()
    learn.fit_one_cycle(epochs, lr)
    timer.stop()

    return learn


def save_artifacts(
    mrf: MiniRocketFeatures,
    learn: Learner,
    X: np.ndarray,
    output_dir: str | Path = "./models",
    tag: str = "name_that_move",
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
) -> None:
    """Persist the feature extractor, learner, and input-shape metadata.

    Saved files
    -----------
    ``<output_dir>/MRF-<tag>.pt``          – ``MiniRocketFeatures`` state dict
    ``<output_dir>/MRL-<tag>.pkl``         – exported ``fastai`` Learner
    ``<output_dir>/input_shape-<tag>.pt``  – ``{n_channels, n_timesteps}`` dict

    Parameters
    ----------
    mrf : fitted ``MiniRocketFeatures``
        Feature extractor whose state dictionary will be saved.
    learn : trained ``fastai`` ``Learner``
        Classification learner to export.
    X : ndarray, shape ``(N, n_channels, n_timesteps)``
        Full feature-extraction input used to record the model input shape.
    output_dir : str or Path
        Directory to write artifacts into; created when absent.
    tag : str
        Filename suffix used to distinguish model versions.
    config : IMUWindowConfig
        Sampling, duration, and channel-order metadata to validate and persist.

    """
    X = validate_windows(X, config=config)
    output_dir = Path(validate_path(output_dir, name="output_dir"))
    tag = validate_tag(tag)
    if not callable(getattr(mrf, "state_dict", None)):
        raise TypeError("mrf must provide a callable state_dict method")
    if not callable(getattr(learn, "export", None)):
        raise TypeError("learn must provide a callable export method")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(mrf.state_dict(), output_dir / f"MRF-{tag}.pt")
    learn.export(output_dir / f"MRL-{tag}.pkl")
    metadata = {
        "n_channels": X.shape[1],
        "n_timesteps": X.shape[2],
        "sample_rate_hz": config.sample_rate_hz,
        "window_duration_s": config.window_duration_s,
        "channel_names": list(config.channel_names),
    }
    torch.save(metadata, output_dir / f"input_shape-{tag}.pt")
    print(f"Artifacts saved to '{output_dir}/' with tag '{tag}'")
