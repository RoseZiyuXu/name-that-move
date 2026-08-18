import pickle

import numpy as np

from minirocket_on_the_fly.realtime.osc_receiver import osc_channel_paths
from minirocket_on_the_fly.realtime.remote_client import (
    RemoteModelClient,
    RemotePrediction,
)
from minirocket_on_the_fly.realtime.touchdesigner import TouchDesignerClient
from minirocket_on_the_fly.realtime.window_buffer import (
    CompletedWindow,
    WindowDiagnostics,
)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"result": [["circle"], [0.9]]}


class FakeSession:
    def __init__(self):
        self.batch = None

    def post(self, url, *, files, timeout):
        assert url == "http://model.example/process"
        assert timeout == 2.0
        self.batch = pickle.load(files["file"][1])
        return FakeResponse()


class FakeOSCClient:
    def __init__(self):
        self.messages = []

    def send_message(self, path, value):
        self.messages.append((path, value))


def make_window():
    return CompletedWindow(
        data=np.ones((6, 96), dtype=np.float32),
        diagnostics=WindowDiagnostics(0.0, 2.0, 576, 0.01),
    )


def test_osc_paths_match_existing_phone_sender_convention():
    paths = osc_channel_paths(2)

    assert paths["acc_x"] == "/m/2/acc/x"
    assert paths["gyro_z"] == "/m/2/gyro/z"


def test_remote_client_uploads_batch_without_temporary_file():
    session = FakeSession()
    client = RemoteModelClient(
        "http://model.example/process", timeout_s=2, session=session
    )

    result = client.predict(make_window())

    assert result == RemotePrediction(label="circle", confidence=0.9)
    assert session.batch.shape == (1, 6, 96)


def test_touchdesigner_client_sends_label_and_confidence():
    osc_client = FakeOSCClient()
    client = TouchDesignerClient(client=osc_client, base_path="/python")

    client.send(RemotePrediction("circle", 0.9))

    assert osc_client.messages == [
        ("/python/label", "circle"),
        ("/python/confidence", 0.9),
    ]
