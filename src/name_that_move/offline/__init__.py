"""Offline dataset preparation and model-training workflow."""

from name_that_move.offline.data import (
    augment_segments,
    load_segments,
    make_dataset,
    make_session_dataset,
)
from name_that_move.offline.evaluation import SessionEvaluation, evaluate_session
from name_that_move.offline.training import (
    extract_features,
    save_artifacts,
    train,
)

__all__ = [
    "SessionEvaluation",
    "augment_segments",
    "extract_features",
    "evaluate_session",
    "load_segments",
    "make_dataset",
    "make_session_dataset",
    "save_artifacts",
    "train",
]
