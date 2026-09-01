"""Name That Move: IMU classification for dance and performance."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DEFAULT_CHANNEL_NAMES": ("name_that_move.preprocessing", "DEFAULT_CHANNEL_NAMES"),
    "DEFAULT_IMU_CONFIG": ("name_that_move.preprocessing", "DEFAULT_IMU_CONFIG"),
    "IMUWindowConfig": ("name_that_move.preprocessing", "IMUWindowConfig"),
    "SessionEvaluation": ("name_that_move.offline.evaluation", "SessionEvaluation"),
    "augment_segments": ("name_that_move.offline.data", "augment_segments"),
    "discover_recording_sessions": (
        "name_that_move.offline.data",
        "discover_recording_sessions",
    ),
    "evaluate_session": ("name_that_move.offline.evaluation", "evaluate_session"),
    "extract_features": ("name_that_move.offline.training", "extract_features"),
    "load_model": ("name_that_move.infer", "load_model"),
    "load_segment": ("name_that_move.window_io", "load_segment"),
    "load_segments": ("name_that_move.offline.data", "load_segments"),
    "load_segments_batch": ("name_that_move.window_io", "load_segments_batch"),
    "make_dataset": ("name_that_move.offline.data", "make_dataset"),
    "make_session_dataset": ("name_that_move.offline.data", "make_session_dataset"),
    "make_windows": ("name_that_move.preprocessing", "make_windows"),
    "predict": ("name_that_move.infer", "predict"),
    "save_artifacts": ("name_that_move.offline.training", "save_artifacts"),
    "train": ("name_that_move.offline.training", "train"),
    "validate_windows": ("name_that_move.preprocessing", "validate_windows"),
}

__all__ = [
    "DEFAULT_CHANNEL_NAMES",
    "DEFAULT_IMU_CONFIG",
    "IMUWindowConfig",
    "SessionEvaluation",
    "augment_segments",
    "discover_recording_sessions",
    "evaluate_session",
    "extract_features",
    "load_model",
    "load_segment",
    "load_segments",
    "load_segments_batch",
    "make_dataset",
    "make_session_dataset",
    "make_windows",
    "predict",
    "save_artifacts",
    "train",
    "validate_windows",
]


def __getattr__(name: str) -> Any:
    """Load public objects only when they are first requested."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes, including lazily exported public objects."""
    return sorted(set(globals()) | set(__all__))
