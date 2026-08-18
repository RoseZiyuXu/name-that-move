import json
import pickle

import numpy as np
import pytest

from minirocket_on_the_fly.realtime.recorder import AsyncWindowRecorder
from minirocket_on_the_fly.realtime.window_buffer import (
    CompletedWindow,
    WindowDiagnostics,
)


def test_async_recorder_saves_data_and_metadata(tmp_path):
    window = CompletedWindow(
        data=np.ones((6, 96), dtype=np.float32),
        diagnostics=WindowDiagnostics(
            started_at=1.0,
            ended_at=3.0,
            osc_message_count=576,
            max_channel_age_s=0.01,
        ),
    )
    recorder = AsyncWindowRecorder(
        tmp_path,
        label="line",
        session="session_01",
        imu_id=2,
    )

    assert recorder.submit(window)
    recorder.close()

    data_files = list(recorder.recording_dir.glob("*.pkl"))
    metadata_files = list(recorder.recording_dir.glob("*.json"))
    assert len(data_files) == 1
    assert len(metadata_files) == 1
    with data_files[0].open("rb") as file:
        np.testing.assert_array_equal(pickle.load(file), window.data)
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["label"] == "line"
    assert metadata["shape"] == [6, 96]
    assert metadata["osc_message_count"] == 576


def test_async_recorder_rejects_window_that_does_not_match_config(tmp_path):
    window = CompletedWindow(
        data=np.ones((6, 95), dtype=np.float32),
        diagnostics=WindowDiagnostics(1.0, 3.0, 570, 0.01),
    )
    recorder = AsyncWindowRecorder(
        tmp_path,
        label="line",
        session="session_01",
        imu_id=2,
    )

    with pytest.raises(ValueError, match=r"Expected window shape \(6, 96\)"):
        recorder.submit(window)
    recorder.close()
