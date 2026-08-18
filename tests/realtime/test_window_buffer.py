import numpy as np
import pytest

from minirocket_on_the_fly.preprocessing import IMUWindowConfig
from minirocket_on_the_fly.realtime.window_buffer import LatestValueWindowBuffer


def test_window_buffer_waits_for_every_channel():
    config = IMUWindowConfig(sample_rate_hz=2, window_duration_s=1)
    buffer = LatestValueWindowBuffer(config)

    for channel in config.channel_names[:-1]:
        buffer.update(channel, 1.0, received_at=0.0)

    assert buffer.sample(sampled_at=0.0) is None


def test_window_buffer_creates_channel_first_window_and_diagnostics():
    config = IMUWindowConfig(sample_rate_hz=2, window_duration_s=1)
    buffer = LatestValueWindowBuffer(config)

    for index, channel in enumerate(config.channel_names):
        buffer.update(channel, float(index), received_at=0.0)
    assert buffer.sample(sampled_at=0.0) is None

    for index, channel in enumerate(config.channel_names):
        buffer.update(channel, float(index + 10), received_at=0.5)
    result = buffer.sample(sampled_at=0.5)

    assert result is not None
    assert result.data.shape == (6, 2)
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data[:, 0], np.arange(6))
    np.testing.assert_array_equal(result.data[:, 1], np.arange(10, 16))
    assert result.diagnostics.osc_message_count == 6
    assert result.diagnostics.max_channel_age_s == 0.0


def test_window_buffer_rejects_invalid_channel_values():
    buffer = LatestValueWindowBuffer()

    with pytest.raises(KeyError, match="Unknown IMU channel"):
        buffer.update("wrong_axis", 1.0)
    with pytest.raises(ValueError, match="finite"):
        buffer.update("acc_x", np.nan)

