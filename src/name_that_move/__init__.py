"""Name That Move: IMU classification for dance and performance."""

from name_that_move.infer import (
    load_model,
    load_segment,
    load_segments_batch,
    predict,
)
from name_that_move.offline import (
    augment_segments,
    extract_features,
    load_segments,
    make_dataset,
    save_artifacts,
    train,
)
from name_that_move.preprocessing import (
    DEFAULT_CHANNEL_NAMES,
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    make_windows,
    validate_windows,
)

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
