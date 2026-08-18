"""Real-time OSC acquisition, recording, inference, and media output."""

from minirocket_on_the_fly.realtime.inference_worker import InferenceWorker
from minirocket_on_the_fly.realtime.osc_receiver import OSCReceiver, osc_channel_paths
from minirocket_on_the_fly.realtime.pipeline import RealtimePipeline
from minirocket_on_the_fly.realtime.recorder import AsyncWindowRecorder
from minirocket_on_the_fly.realtime.remote_client import (
    RemoteModelClient,
    RemotePrediction,
)
from minirocket_on_the_fly.realtime.touchdesigner import TouchDesignerClient
from minirocket_on_the_fly.realtime.window_buffer import (
    CompletedWindow,
    LatestValueWindowBuffer,
    WindowDiagnostics,
)

__all__ = [
    "AsyncWindowRecorder",
    "CompletedWindow",
    "InferenceWorker",
    "LatestValueWindowBuffer",
    "OSCReceiver",
    "RealtimePipeline",
    "RemoteModelClient",
    "RemotePrediction",
    "TouchDesignerClient",
    "WindowDiagnostics",
    "osc_channel_paths",
]
