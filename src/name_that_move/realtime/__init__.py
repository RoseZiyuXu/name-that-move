"""Real-time OSC acquisition, recording, inference, and media output."""

from name_that_move.realtime.inference_worker import InferenceWorker
from name_that_move.realtime.local_model import LocalModelPredictor
from name_that_move.realtime.osc_receiver import (
    DEFAULT_IMU_ID,
    DEFAULT_OSC_IP,
    DEFAULT_OSC_PORT,
    OSCReceiver,
    osc_channel_paths,
)
from name_that_move.realtime.pipeline import RealtimePipeline
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.predictor import build_predictor
from name_that_move.realtime.recorder import AsyncWindowRecorder
from name_that_move.realtime.remote_client import (
    RemoteModelClient,
    RemoteInferenceError,
    RemotePrediction,
)
from name_that_move.realtime.touchdesigner import TouchDesignerClient
from name_that_move.realtime.window_buffer import (
    CompletedWindow,
    LatestValueWindowBuffer,
    WindowDiagnostics,
)

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
    "RemoteModelClient",
    "RemoteInferenceError",
    "RemotePrediction",
    "TouchDesignerClient",
    "WindowDiagnostics",
    "build_predictor",
    "osc_channel_paths",
]
