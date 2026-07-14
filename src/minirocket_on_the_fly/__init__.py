"""MiniRocket On the Fly: time-series motion classification."""

from minirocket_on_the_fly.data import augment_segments, load_segments, make_dataset
from minirocket_on_the_fly.infer import (
    load_model,
    load_segment,
    load_segments_batch,
    predict,
)
from minirocket_on_the_fly.train import extract_features, save_artifacts, train

__all__ = [
    # data
    "load_segments",
    "augment_segments",
    "make_dataset",
    # train
    "extract_features",
    "train",
    "save_artifacts",
    # infer
    "load_segment",
    "load_segments_batch",
    "load_model",
    "predict",
]
