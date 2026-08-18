import numpy as np
import pytest

from minirocket_on_the_fly.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    make_windows,
    validate_windows,
)


def test_default_config_matches_performance_pipeline():
    assert DEFAULT_IMU_CONFIG.sample_rate_hz == 48.0
    assert DEFAULT_IMU_CONFIG.window_duration_s == 2.0
    assert DEFAULT_IMU_CONFIG.n_channels == 6
    assert DEFAULT_IMU_CONFIG.n_timesteps == 96


def test_make_windows_creates_non_overlapping_model_batches():
    samples = np.arange(192 * 6).reshape(192, 6)

    result = make_windows(samples)

    assert result.shape == (2, 6, 96)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0], samples[:96].T)
    np.testing.assert_array_equal(result[1], samples[96:].T)


def test_make_windows_supports_overlapping_stride():
    samples = np.arange(192 * 6).reshape(192, 6)

    result = make_windows(samples, stride=48)

    assert result.shape == (3, 6, 96)
    np.testing.assert_array_equal(result[1], samples[48:144].T)


def test_validate_windows_adds_batch_dimension_and_casts_to_float32():
    window = np.ones((6, 96), dtype=np.float64)

    result = validate_windows(window, config=DEFAULT_IMU_CONFIG)

    assert result.shape == (1, 6, 96)
    assert result.dtype == np.float32


@pytest.mark.parametrize("shape", [(5, 96), (6, 95)])
def test_validate_windows_rejects_shape_incompatible_with_config(shape):
    with pytest.raises(ValueError, match="Expected window shape"):
        validate_windows(np.ones(shape), config=DEFAULT_IMU_CONFIG)


def test_preprocessing_rejects_non_finite_values():
    samples = np.ones((96, 6))
    samples[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        make_windows(samples)


def test_config_requires_a_whole_number_of_samples():
    with pytest.raises(ValueError, match="whole number"):
        IMUWindowConfig(sample_rate_hz=25, window_duration_s=0.5)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("sample_rate_hz", "48", TypeError),
        ("sample_rate_hz", np.inf, ValueError),
        ("window_duration_s", True, TypeError),
        ("window_duration_s", np.nan, ValueError),
    ],
)
def test_config_rejects_invalid_numeric_settings(field, value, error_type):
    kwargs = {field: value}

    with pytest.raises(error_type, match=field):
        IMUWindowConfig(**kwargs)


def test_config_rejects_invalid_channel_names():
    with pytest.raises(TypeError, match="channel_names"):
        IMUWindowConfig(channel_names=("acc_x", ""))
