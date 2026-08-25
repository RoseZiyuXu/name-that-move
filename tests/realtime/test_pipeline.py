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
