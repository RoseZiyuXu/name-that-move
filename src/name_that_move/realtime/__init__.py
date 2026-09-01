"""Real-time OSC acquisition, recording, inference, and media output."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AsyncWindowRecorder": ("name_that_move.realtime.recorder", "AsyncWindowRecorder"),
    "CompletedWindow": ("name_that_move.realtime.window_buffer", "CompletedWindow"),
    "DEFAULT_IMU_ID": ("name_that_move.realtime.osc_receiver", "DEFAULT_IMU_ID"),
    "DEFAULT_OSC_IP": ("name_that_move.realtime.osc_receiver", "DEFAULT_OSC_IP"),
    "DEFAULT_OSC_PORT": ("name_that_move.realtime.osc_receiver", "DEFAULT_OSC_PORT"),
    "InferenceWorker": ("name_that_move.realtime.inference_worker", "InferenceWorker"),
    "LatestValueWindowBuffer": (
        "name_that_move.realtime.window_buffer",
        "LatestValueWindowBuffer",
    ),
    "LocalModelPredictor": ("name_that_move.realtime.local_model", "LocalModelPredictor"),
    "OSCReceiver": ("name_that_move.realtime.osc_receiver", "OSCReceiver"),
    "Prediction": ("name_that_move.realtime.prediction", "Prediction"),
    "RealtimePipeline": ("name_that_move.realtime.pipeline", "RealtimePipeline"),
    "RemoteInferenceError": (
        "name_that_move.realtime.remote_client",
        "RemoteInferenceError",
    ),
    "RemoteModelClient": ("name_that_move.realtime.remote_client", "RemoteModelClient"),
    "RemotePrediction": ("name_that_move.realtime.remote_client", "RemotePrediction"),
    "TouchDesignerClient": (
        "name_that_move.realtime.touchdesigner",
        "TouchDesignerClient",
    ),
    "WindowDiagnostics": ("name_that_move.realtime.window_buffer", "WindowDiagnostics"),
    "build_predictor": ("name_that_move.realtime.predictor", "build_predictor"),
    "osc_channel_paths": ("name_that_move.realtime.osc_receiver", "osc_channel_paths"),
}

__all__ = [
    "DEFAULT_IMU_ID",
    "DEFAULT_OSC_IP",
    "DEFAULT_OSC_PORT",
    "AsyncWindowRecorder",
    "CompletedWindow",
    "InferenceWorker",
    "LatestValueWindowBuffer",
    "LocalModelPredictor",
    "OSCReceiver",
    "Prediction",
    "RealtimePipeline",
    "RemoteInferenceError",
    "RemoteModelClient",
    "RemotePrediction",
    "TouchDesignerClient",
    "WindowDiagnostics",
    "build_predictor",
    "osc_channel_paths",
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
