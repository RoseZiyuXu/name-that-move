import pickle

import numpy as np
import pytest
import requests

from name_that_move.realtime.osc_receiver import (
    DEFAULT_IMU_ID,
    OSCReceiver,
    osc_channel_paths,
)
from name_that_move.realtime.remote_client import (
    RemoteInferenceError,
    RemoteModelClient,
    RemotePrediction,
)
from name_that_move.realtime.touchdesigner import TouchDesignerClient
from name_that_move.realtime.window_buffer import (
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
        assert url == "http://model.test/process"
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


def test_default_imu_id_is_one():
    paths = osc_channel_paths(DEFAULT_IMU_ID)

    assert DEFAULT_IMU_ID == 1
    assert paths["acc_x"] == "/m/1/acc/x"
    assert paths["gyro_z"] == "/m/1/gyro/z"


def test_osc_receiver_unwraps_pythonosc_mapped_channel_argument():
    received = []
    receiver = OSCReceiver(
        lambda channel, value: received.append((channel, value)),
        imu_id=7,
    )

    receiver._handle_message("/m/7/acc/x", ["acc_x"], 0.5)

    assert received == [("acc_x", 0.5)]


def test_osc_receiver_accepts_direct_mapped_channel_argument():
    received = []
    receiver = OSCReceiver(
        lambda channel, value: received.append((channel, value)),
        imu_id=7,
    )

    receiver._handle_message("/m/7/gyro/z", "gyro_z", -0.25)

    assert received == [("gyro_z", -0.25)]


def test_remote_client_uploads_batch_without_temporary_file():
    session = FakeSession()
    client = RemoteModelClient(
        "http://model.test/process", http_timeout_s=2, session=session
    )

    result = client.predict(make_window())

    assert result == RemotePrediction(label="circle", confidence=0.9)
    assert session.batch.shape == (1, 6, 96)


def test_remote_client_rejects_documentation_placeholder():
    with pytest.raises(ValueError, match="documentation placeholder"):
        RemoteModelClient(
            "https://your-model-server.example/process",
            session=FakeSession(),
        )


def test_remote_client_simplifies_request_failures():
    class FailingSession:
        def post(self, *args, **kwargs):
            raise requests.ConnectionError("very long low-level traceback")

    client = RemoteModelClient(
        "https://model.test/process",
        http_timeout_s=2,
        session=FailingSession(),
    )

    with pytest.raises(RemoteInferenceError, match="model.test"):
        client.predict(make_window())


def test_touchdesigner_client_sends_label_and_confidence():
    osc_client = FakeOSCClient()
    client = TouchDesignerClient(client=osc_client, base_path="/python")

    client.send(RemotePrediction("circle", 0.9))

    assert osc_client.messages == [
        ("/python/label", "circle"),
        ("/python/confidence", 0.9),
    ]
