"""Offline dataset preparation and model-training workflow."""

from minirocket_on_the_fly.offline.data import (
    augment_segments,
    load_segments,
    make_dataset,
)
from minirocket_on_the_fly.offline.training import (
    extract_features,
    save_artifacts,
    train,
)

__all__ = [
    "augment_segments",
    "extract_features",
    "load_segments",
    "make_dataset",
    "save_artifacts",
    "train",
]
