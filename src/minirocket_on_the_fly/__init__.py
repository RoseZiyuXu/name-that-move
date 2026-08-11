"""MiniRocket On the Fly: time-series motion classification."""

from minirocket_on_the_fly.data import augment_segments, load_segments, make_dataset
from minirocket_on_the_fly.infer import (
    load_model,
    load_segment,
    load_segments_batch,
    predict,
)
from minirocket_on_the_fly.preprocessing import (
    DEFAULT_CHANNEL_NAMES,
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    make_windows,
    validate_windows,
)
from minirocket_on_the_fly.train import extract_features, save_artifacts, train

__all__ = [
    "DEFAULT_CHANNEL_NAMES",
    "DEFAULT_IMU_CONFIG",
    "IMUWindowConfig",
    "augment_segments",
    "extract_features",
    "load_model",
    "load_segment",
    "load_segments",
    "load_segments_batch",
    "make_dataset",
    "make_windows",
    "predict",
    "save_artifacts",
    "train",
    "validate_windows",
]
