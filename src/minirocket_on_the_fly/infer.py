"""Inference with a trained MiniRocket model.

An IMU window is one fixed-duration segment of continuous sensor data with
channel-first shape ``(n_channels, n_timesteps)``.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tsai.basics import load_learner
from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures

from minirocket_on_the_fly._compat import default_device, get_minirocket_features
from minirocket_on_the_fly._validation import (
    validate_path,
    validate_positive_int,
    validate_tag,
)
from minirocket_on_the_fly.preprocessing import validate_windows


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
        Loaded numeric window in channel-first format.

    """
    path = Path(validate_path(path, name="path"))
    with path.open("rb") as fh:
        data = pickle.load(fh)

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
    """Load a ``.pkl`` file that already contains a batch of segments.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file.  Name is not restricted.

    Returns
    -------
    ndarray, shape ``(N, n_channels, n_timesteps)``
        Loaded windows as a contiguous ``float32`` batch.

    """
    path = Path(validate_path(path, name="path"))
    with path.open("rb") as fh:
        data = pickle.load(fh)

    return validate_windows(data)


def load_model(
    model_dir: str | Path = "./models",
    tag: str = "minirocket_on_the_fly",
) -> tuple[MiniRocketFeatures, Any]:
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
        Restored feature extractor in evaluation mode.
    learn : loaded ``fastai`` Learner
        Restored classification learner.

    """
    model_dir = Path(validate_path(model_dir, name="model_dir"))
    tag = validate_tag(tag)
    shape_path = model_dir / f"input_shape-{tag}.pt"
    feature_path = model_dir / f"MRF-{tag}.pt"
    learner_path = model_dir / f"MRL-{tag}.pkl"

    missing_paths = [
        path for path in (shape_path, feature_path, learner_path) if not path.is_file()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing saved-model artifact(s): {missing}")

    input_shape = torch.load(shape_path)
    if not isinstance(input_shape, Mapping):
        raise ValueError("input-shape metadata must be a mapping")
    try:
        n_channels = validate_positive_int(
            input_shape["n_channels"], name="n_channels"
        )
        n_timesteps = validate_positive_int(
            input_shape["n_timesteps"], name="n_timesteps"
        )
    except KeyError as error:
        raise ValueError(
            "input-shape metadata must contain n_channels and n_timesteps"
        ) from error
    print(f"Model input shape: {input_shape}")

    mrf = (
        MiniRocketFeatures(n_channels, n_timesteps)
        .to(default_device())
        .float()
    )
    mrf.load_state_dict(torch.load(feature_path))
    mrf.eval()

    learn = load_learner(learner_path, cpu=False)

    return mrf, learn


def predict(
    X: np.ndarray,
    mrf: MiniRocketFeatures,
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
        Predicted class probabilities for every input window.
    preds  : list of predicted class labels (length N)
        Predicted label for every input window.

    """
    X = validate_windows(X)
    chunksize = validate_positive_int(chunksize, name="chunksize")
    if not callable(getattr(learn, "get_X_preds", None)):
        raise TypeError("learn must provide a callable get_X_preds method")

    X_feat = get_minirocket_features(X, mrf, chunksize=chunksize, to_np=True)
    probas, _, preds_tensor = learn.get_X_preds(X_feat)

    return np.array(probas), preds_tensor.tolist()
