"""MiniRocket feature extraction and linear-head training."""

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
from tsai.models.utils import build_ts_model, default_device, get_minirocket_features


def extract_features(
    X: np.ndarray,
    splits: tuple[np.ndarray, np.ndarray],
    chunksize: int = 32,
) -> tuple[np.ndarray, "MiniRocketFeatures"]:
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

    Returns
    -------
    X_feat : ndarray, shape ``(N, n_features)``
    mrf    : fitted ``MiniRocketFeatures`` instance (on ``default_device()``)
    """
    X = X.astype(np.float32)
    X_train = X[splits[0]]

    mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device()).float()
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
) -> "Learner":
    """Train a ``MiniRocketHead`` linear classifier on pre-extracted features.

    If ``lr`` is ``None`` the learning-rate finder is run automatically and
    its ``valley`` suggestion is used.

    Parameters
    ----------
    X_feat : ndarray, shape ``(N, n_features)``
    y : ndarray of str labels, shape ``(N,)``
    splits : ``(train_indices, val_indices)`` tuple
    epochs : number of ``fit_one_cycle`` epochs
    batch_size : DataLoader batch size
    lr : fixed learning rate; ``None`` → auto-detect via lr_find

    Returns
    -------
    learn : trained ``fastai`` ``Learner``
    """
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
    learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraph())

    timer.start()
    learn.fit_one_cycle(epochs, lr)
    timer.stop()

    return learn


def save_artifacts(
    mrf: "MiniRocketFeatures",
    learn: "Learner",
    X: np.ndarray,
    output_dir: str | Path = "./models",
    tag: str = "motion_clf",
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
    learn : trained ``fastai`` ``Learner``
    X : the full feature-extraction input, shape ``(N, n_channels, n_timesteps)``
    output_dir : directory to write artefacts into (created if absent)
    tag : filename suffix to distinguish model versions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(mrf.state_dict(), output_dir / f"MRF-{tag}.pt")
    learn.export(output_dir / f"MRL-{tag}.pkl")
    torch.save(
        {"n_channels": X.shape[1], "n_timesteps": X.shape[2]},
        output_dir / f"input_shape-{tag}.pt",
    )
    print(f"Artifacts saved to '{output_dir}/' with tag '{tag}'")
