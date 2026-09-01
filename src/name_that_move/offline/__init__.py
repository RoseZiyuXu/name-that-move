"""Offline dataset preparation and model-training workflow."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "SessionEvaluation": ("name_that_move.offline.evaluation", "SessionEvaluation"),
    "augment_segments": ("name_that_move.offline.data", "augment_segments"),
    "discover_recording_sessions": (
        "name_that_move.offline.data",
        "discover_recording_sessions",
    ),
    "evaluate_session": ("name_that_move.offline.evaluation", "evaluate_session"),
    "extract_features": ("name_that_move.offline.training", "extract_features"),
    "load_segments": ("name_that_move.offline.data", "load_segments"),
    "make_dataset": ("name_that_move.offline.data", "make_dataset"),
    "make_session_dataset": ("name_that_move.offline.data", "make_session_dataset"),
    "save_artifacts": ("name_that_move.offline.training", "save_artifacts"),
    "train": ("name_that_move.offline.training", "train"),
}

__all__ = [
    "SessionEvaluation",
    "augment_segments",
    "discover_recording_sessions",
    "evaluate_session",
    "extract_features",
    "load_segments",
    "make_dataset",
    "make_session_dataset",
    "save_artifacts",
    "train",
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
