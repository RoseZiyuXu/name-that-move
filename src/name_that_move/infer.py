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

from name_that_move._compat import default_device, get_minirocket_features
from name_that_move._validation import (
    validate_path,
    validate_positive_int,
    validate_tag,
)
from name_that_move.preprocessing import IMUWindowConfig, validate_windows


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
    tag: str = "name_that_move",
    *,
    expected_config: IMUWindowConfig | None = None,
    cpu: bool = True,
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
    expected_config:
        Optional runtime configuration. When provided, the saved channel
        count, timestep count, sampling rate, duration, and channel order must
        all match before the model is constructed.
    cpu:
        Load the feature extractor and learner on CPU. This stable,
        cross-platform default is appropriate for tests and small real-time
        models. Set to ``False`` to use the available accelerated device.

    Returns
    -------
    mrf   : ``MiniRocketFeatures`` with loaded weights, on ``default_device()``
        Restored feature extractor in evaluation mode.
    learn : loaded ``fastai`` Learner
        Restored classification learner.

    """
    model_dir = Path(validate_path(model_dir, name="model_dir"))
    tag = validate_tag(tag)
    if not isinstance(cpu, bool):
        raise TypeError("cpu must be a boolean")
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
        raise TypeError("input-shape metadata must be a mapping")
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
    if expected_config is not None:
        _validate_saved_config(
            input_shape,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            expected_config=expected_config,
        )
    print(f"Model input shape: {input_shape}")

    device = torch.device("cpu") if cpu else default_device()
    mrf = MiniRocketFeatures(n_channels, n_timesteps).to(device).float()
    mrf.load_state_dict(torch.load(feature_path, map_location=device))
    mrf.eval()
    mrf._minirocket_input_shape = (n_channels, n_timesteps)

    learn = load_learner(learner_path, cpu=cpu)

    return mrf, learn


def predict(
    X: np.ndarray,
    mrf: MiniRocketFeatures,
    learn: Any,
    chunksize: int = 32,
    *,
    config: IMUWindowConfig | None = None,
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
    config:
        Optional runtime configuration. When provided, input windows must
        match its channel count and timestep count.

    Returns
    -------
    probas : ndarray, shape ``(N, n_classes)`` — class probabilities
        Predicted class probabilities for every input window.
    preds  : list of predicted class labels (length N)
        Predicted label for every input window.

    """
    X = validate_windows(X, config=config)
    model_shape = getattr(mrf, "_minirocket_input_shape", None)
    received_shape = tuple(X.shape[1:])
    if model_shape is not None and tuple(model_shape) != received_shape:
        raise ValueError(
            f"Model expects window shape {tuple(model_shape)}; received "
            f"{received_shape}. Use the same IMUWindowConfig for recording, "
            "training, and inference, or load a compatible model."
        )
    chunksize = validate_positive_int(chunksize, name="chunksize")
    if not callable(getattr(learn, "get_X_preds", None)):
        raise TypeError("learn must provide a callable get_X_preds method")

    X_feat = get_minirocket_features(X, mrf, chunksize=chunksize, to_np=True)
    probas, _, preds_tensor = learn.get_X_preds(X_feat)

    return np.array(probas), preds_tensor.tolist()


def _validate_saved_config(
    metadata: Mapping,
    *,
    n_channels: int,
    n_timesteps: int,
    expected_config: IMUWindowConfig,
) -> None:
    """Fail fast when saved-model and runtime IMU configurations differ."""
    required = ("sample_rate_hz", "window_duration_s", "channel_names")
    missing = [name for name in required if name not in metadata]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            f"Saved model metadata is missing {fields}. Re-save the model "
            "with save_artifacts(..., config=config) before configuration "
            "matching can be verified."
        )

    saved_shape = (n_channels, n_timesteps)
    expected_shape = (expected_config.n_channels, expected_config.n_timesteps)
    if saved_shape != expected_shape:
        raise ValueError(
            f"Saved model expects window shape {saved_shape}, but runtime "
            f"configuration produces {expected_shape}. Use a matching "
            "IMUWindowConfig or load a compatible model."
        )

    channel_names = metadata["channel_names"]
    if isinstance(channel_names, str):
        raise TypeError("Saved channel_names must be a sequence of channel names")
    try:
        saved_rate = float(metadata["sample_rate_hz"])
        saved_duration = float(metadata["window_duration_s"])
        saved_channels = tuple(channel_names)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Saved sample_rate_hz and window_duration_s must be numeric, and "
            "channel_names must be iterable"
        ) from error

    mismatches = []
    if not np.isclose(saved_rate, expected_config.sample_rate_hz):
        mismatches.append(
            f"sample_rate_hz saved={saved_rate}, "
            f"runtime={expected_config.sample_rate_hz}"
        )
    if not np.isclose(saved_duration, expected_config.window_duration_s):
        mismatches.append(
            f"window_duration_s saved={saved_duration}, "
            f"runtime={expected_config.window_duration_s}"
        )
    if saved_channels != expected_config.channel_names:
        mismatches.append(
            f"channel_names saved={saved_channels}, "
            f"runtime={expected_config.channel_names}"
        )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"Saved model configuration does not match runtime configuration: "
            f"{details}. Use the same IMUWindowConfig for recording, training, "
            "and inference, or load a compatible model."
        )
