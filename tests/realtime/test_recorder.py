import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pytest

from name_that_move.realtime.recorder import AsyncWindowRecorder
from name_that_move.realtime.window_buffer import (
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
    assert recorder.submitted_count == 1
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


def test_async_recorder_uses_readable_local_timestamp_and_avoids_collisions(
    tmp_path, monkeypatch
):
    captured_at = datetime(
        2026, 9, 1, 17, 18, 39, 720_000, tzinfo=timezone.utc
    )

    class FixedDateTime:
        @classmethod
        def now(cls, tz):
            return captured_at.astimezone(tz)

    monkeypatch.setattr("name_that_move.realtime.recorder.datetime", FixedDateTime)
    window = CompletedWindow(
        data=np.ones((6, 96), dtype=np.float32),
        diagnostics=WindowDiagnostics(1.0, 3.0, 576, 0.01),
    )
    recorder = AsyncWindowRecorder(
        tmp_path,
        label="water",
        session="water_session_01",
        imu_id=7,
    )

    assert recorder.submit(window)
    assert recorder.submit(window)
    assert recorder.submitted_count == 2
    recorder.close()

    captured_at_local = captured_at.astimezone()
    local_timestamp = (
        f"{captured_at_local:%Y%m%d_%H%M%S}_"
        f"{captured_at_local.microsecond // 10_000:02d}"
    )
    stems = sorted(path.stem for path in recorder.recording_dir.glob("*.pkl"))
    assert stems == [
        f"imu7_{local_timestamp}",
        f"imu7_{local_timestamp}_02",
    ]
