"""motion-clf: Time-series motion classification with MiniRocket."""

from motion_clf.data import augment_segments, load_segments, make_dataset
from motion_clf.infer import load_model, load_segment, load_segments_batch, predict
from motion_clf.train import extract_features, save_artifacts, train

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
