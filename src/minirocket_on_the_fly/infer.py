"""Inference with a trained MiniRocket model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tsai.basics import load_learner
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures

from minirocket_on_the_fly._compat import default_device, get_minirocket_features


def load_segment(path: str | Path) -> np.ndarray:
    """Load a single ``.pkl`` segment file.

    The file may contain either a bare ``np.ndarray`` of shape
    ``(n_channels, n_timesteps)`` or a sequence whose first element is one.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file.  Name is not restricted.

    Returns
    -------
    ndarray, shape ``(n_channels, n_timesteps)``
    """
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)

    if isinstance(data, np.ndarray):
        return data
    # sequence convention used during training: segment[0] is the array
    return data[0]


def load_segments_batch(path: str | Path) -> np.ndarray:
    """Load a ``.pkl`` file that already contains a batch of segments.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file.  Name is not restricted.

    Returns
    -------
    ndarray, shape ``(N, n_channels, n_timesteps)``
    """
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)

    arr = np.array(data)
    if arr.ndim == 2:
        # Single sample stored without batch dimension — add it.
        arr = arr[np.newaxis, ...]
    return arr


def load_model(
    model_dir: str | Path = "./models",
    tag: str = "minirocket_on_the_fly",
) -> tuple["MiniRocketFeatures", Any]:
    """Load the feature extractor and the trained learner from disk.

    Expects the three files written by ``save_artifacts``::

        <model_dir>/MRF-<tag>.pt
        <model_dir>/MRL-<tag>.pkl
        <model_dir>/input_shape-<tag>.pt

    Parameters
    ----------
    model_dir:
        Directory that contains the saved artefacts.
    tag:
        The tag string used when the artefacts were saved.

    Returns
    -------
    mrf   : ``MiniRocketFeatures`` with loaded weights, on ``default_device()``
    learn : loaded ``fastai`` Learner
    """
    model_dir = Path(model_dir)

    input_shape = torch.load(model_dir / f"input_shape-{tag}.pt")
    print(f"Model input shape: {input_shape}")

    mrf = (
        MiniRocketFeatures(input_shape["n_channels"], input_shape["n_timesteps"])
        .to(default_device())
        .float()
    )
    mrf.load_state_dict(torch.load(model_dir / f"MRF-{tag}.pt"))
    mrf.eval()

    learn = load_learner(model_dir / f"MRL-{tag}.pkl", cpu=False)

    return mrf, learn


def predict(
    X: np.ndarray,
    mrf: "MiniRocketFeatures",
    learn: Any,
    chunksize: int = 32,
) -> tuple[np.ndarray, list]:
    """Run inference on a batch of segments.

    Parameters
    ----------
    X:
        Float32 array of shape ``(N, n_channels, n_timesteps)``.
    mrf:
        Fitted ``MiniRocketFeatures`` (from ``load_model``).
    learn:
        Trained ``fastai`` Learner (from ``load_model``).
    chunksize:
        Batch size used during feature extraction.

    Returns
    -------
    probas : ndarray, shape ``(N, n_classes)`` — class probabilities
    preds  : list of predicted class labels (length N)
    """
    X = np.array(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[np.newaxis, ...]  # single sample → batch of 1

    X_feat = get_minirocket_features(X, mrf, chunksize=chunksize, to_np=True)
    probas, _, preds_tensor = learn.get_X_preds(X_feat)

    return np.array(probas), preds_tensor.tolist()
