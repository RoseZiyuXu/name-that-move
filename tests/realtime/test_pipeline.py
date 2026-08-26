import pytest

from name_that_move.preprocessing import IMUWindowConfig
from name_that_move.realtime.pipeline import RealtimePipeline


class StubRecorder:
    def __init__(self, config):
        self.config = config


def test_pipeline_rejects_mismatched_recorder_configuration():
    pipeline_config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
    recorder_config = IMUWindowConfig(sample_rate_hz=52, window_duration_s=2)

    with pytest.raises(ValueError, match="Recorder configuration must match"):
        RealtimePipeline(
            config=pipeline_config,
            recorder=StubRecorder(recorder_config),
        )


def test_pipeline_rejects_invalid_startup_timeout():
    with pytest.raises(ValueError, match="startup_timeout_s"):
        RealtimePipeline(startup_timeout_s=0)


def test_pipeline_stops_when_no_complete_osc_input_arrives(monkeypatch):
    pipeline = RealtimePipeline(startup_timeout_s=0.01)
    monkeypatch.setattr(pipeline.receiver, "start", lambda: None)
    monkeypatch.setattr(pipeline.receiver, "stop", lambda: None)

    with pytest.raises(TimeoutError, match="No complete six-channel OSC input"):
        pipeline.start()

    assert pipeline._sampling_thread is None
