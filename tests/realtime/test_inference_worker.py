from threading import Event
from time import monotonic

import numpy as np

from minirocket_on_the_fly.realtime.inference_worker import InferenceWorker
from minirocket_on_the_fly.realtime.window_buffer import (
    CompletedWindow,
    WindowDiagnostics,
)


def test_inference_submission_does_not_block_sampling_thread():
    predictor_started = Event()
    allow_predictor_to_finish = Event()
    result_received = Event()
    window = CompletedWindow(
        data=np.ones((6, 96), dtype=np.float32),
        diagnostics=WindowDiagnostics(0.0, 2.0, 576, 0.01),
    )

    def slow_predictor(received_window):
        predictor_started.set()
        allow_predictor_to_finish.wait(timeout=2)
        return received_window.data.shape

    def receive_result(result, received_window):
        assert result == (6, 96)
        assert received_window is window
        result_received.set()

    worker = InferenceWorker(slow_predictor, on_result=receive_result)
    started_at = monotonic()
    assert worker.submit(window)
    elapsed = monotonic() - started_at

    assert elapsed < 0.1
    assert predictor_started.wait(timeout=1)
    allow_predictor_to_finish.set()
    worker.close()
    assert result_received.is_set()

