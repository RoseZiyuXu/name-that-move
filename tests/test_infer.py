import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from minirocket_on_the_fly import infer
from minirocket_on_the_fly.infer import (
    load_model,
    load_segment,
    load_segments_batch,
    predict,
)
from minirocket_on_the_fly.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig


def test_load_segment_accepts_wrapped_array(tmp_path):
    expected = np.ones((6, 96), dtype=np.float32)
    path = tmp_path / "segment.pkl"
    with path.open("wb") as file:
        pickle.dump([expected], file)

    np.testing.assert_array_equal(load_segment(path), expected)


def test_load_segments_batch_adds_batch_dimension(tmp_path):
    expected = np.ones((6, 96), dtype=np.float32)
    path = tmp_path / "segment.pkl"
    with path.open("wb") as file:
        pickle.dump(expected, file)

    result = load_segments_batch(path)

    assert result.shape == (1, 6, 96)


def test_load_segments_batch_rejects_non_finite_values(tmp_path):
    path = tmp_path / "segment.pkl"
    invalid = np.ones((1, 6, 96), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with path.open("wb") as file:
        pickle.dump(invalid, file)

    with pytest.raises(ValueError, match="finite"):
        load_segments_batch(path)


def test_load_model_reports_all_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="MRF-example.pt") as error:
        load_model(tmp_path, tag="example")

    message = str(error.value)
    assert "MRL-example.pkl" in message
    assert "input_shape-example.pt" in message


def test_load_model_rejects_non_boolean_cpu_setting(tmp_path):
    with pytest.raises(TypeError, match="cpu"):
        load_model(tmp_path, tag="example", cpu="yes")


def test_load_model_rejects_invalid_metadata_type(tmp_path, monkeypatch):
    for name in ("input_shape-example.pt", "MRF-example.pt", "MRL-example.pkl"):
        (tmp_path / name).touch()
    monkeypatch.setattr(infer.torch, "load", lambda path: [6, 96])

    with pytest.raises(TypeError, match="metadata must be a mapping"):
        load_model(tmp_path, tag="example")


def test_load_model_rejects_runtime_config_mismatch_before_model_creation(
    tmp_path, monkeypatch
):
    for name in ("input_shape-example.pt", "MRF-example.pt", "MRL-example.pkl"):
        (tmp_path / name).touch()
    metadata = {
        "n_channels": 6,
        "n_timesteps": 104,
        "sample_rate_hz": 52.0,
        "window_duration_s": 2.0,
        "channel_names": list(DEFAULT_IMU_CONFIG.channel_names),
    }
    monkeypatch.setattr(infer.torch, "load", lambda path: metadata)

    with pytest.raises(ValueError, match=r"\(6, 104\).+\(6, 96\)"):
        load_model(
            tmp_path,
            tag="example",
            expected_config=DEFAULT_IMU_CONFIG,
        )


def test_load_model_requires_full_metadata_for_config_check(tmp_path, monkeypatch):
    for name in ("input_shape-example.pt", "MRF-example.pt", "MRL-example.pkl"):
        (tmp_path / name).touch()
    monkeypatch.setattr(
        infer.torch,
        "load",
        lambda path: {"n_channels": 6, "n_timesteps": 96},
    )

    with pytest.raises(ValueError, match="Re-save the model"):
        load_model(
            tmp_path,
            tag="example",
            expected_config=DEFAULT_IMU_CONFIG,
        )


def test_load_model_rejects_rate_mismatch_even_when_shape_matches(
    tmp_path, monkeypatch
):
    for name in ("input_shape-example.pt", "MRF-example.pt", "MRL-example.pkl"):
        (tmp_path / name).touch()
    metadata = {
        "n_channels": 6,
        "n_timesteps": 96,
        "sample_rate_hz": 96.0,
        "window_duration_s": 1.0,
        "channel_names": list(DEFAULT_IMU_CONFIG.channel_names),
    }
    monkeypatch.setattr(infer.torch, "load", lambda path: metadata)

    with pytest.raises(ValueError, match="sample_rate_hz saved=96.0"):
        load_model(
            tmp_path,
            tag="example",
            expected_config=DEFAULT_IMU_CONFIG,
        )


def test_predict_rejects_invalid_input_before_feature_extraction(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("feature extraction should not run")

    monkeypatch.setattr(infer, "get_minirocket_features", fail_if_called)
    invalid = np.ones((2, 6, 96), dtype=np.float32)
    invalid[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        predict(invalid, object(), object())


def test_predict_rejects_non_positive_chunksize():
    with pytest.raises(ValueError, match="chunksize"):
        predict(np.ones((1, 6, 96)), object(), object(), chunksize=0)


def test_predict_rejects_model_shape_mismatch_before_feature_extraction(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("feature extraction should not run")

    monkeypatch.setattr(infer, "get_minirocket_features", fail_if_called)
    model = SimpleNamespace(_minirocket_input_shape=(6, 104))

    with pytest.raises(ValueError, match=r"expects window shape \(6, 104\)"):
        predict(np.ones((1, 6, 96)), model, object())


def test_predict_accepts_custom_config_until_learner_validation():
    config = IMUWindowConfig(sample_rate_hz=52, window_duration_s=2)
    model = SimpleNamespace(_minirocket_input_shape=(6, 104))

    with pytest.raises(TypeError, match="get_X_preds"):
        predict(np.ones((1, 6, 104)), model, object(), config=config)
