import importlib
from pathlib import Path

import numpy as np
import pytest

from minirocket_on_the_fly.offline.training import (
    extract_features,
    save_artifacts,
    train,
)
from minirocket_on_the_fly.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig

train_module = importlib.import_module("minirocket_on_the_fly.offline.training")


def test_extract_features_rejects_invalid_windows_before_model_creation(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("MiniRocket should not be created")

    monkeypatch.setattr(train_module, "MiniRocketFeatures", fail_if_called)
    invalid = np.ones((4, 6, 96), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    splits = (np.array([0, 1, 2]), np.array([3]))

    with pytest.raises(ValueError, match="finite"):
        extract_features(invalid, splits)


def test_extract_features_rejects_overlapping_splits():
    X = np.ones((4, 6, 96), dtype=np.float32)
    splits = (np.array([0, 1, 2]), np.array([2, 3]))

    with pytest.raises(ValueError, match="must not overlap"):
        extract_features(X, splits)


def test_extract_features_rejects_config_mismatch_before_model_creation(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("MiniRocket should not be created")

    monkeypatch.setattr(train_module, "MiniRocketFeatures", fail_if_called)
    config = IMUWindowConfig(sample_rate_hz=52, window_duration_s=2)
    X = np.ones((4, 6, 96), dtype=np.float32)
    splits = (np.array([0, 1, 2]), np.array([3]))

    with pytest.raises(ValueError, match=r"Expected window shape \(6, 104\)"):
        extract_features(X, splits, config=config)


def test_train_rejects_mismatched_labels_before_dataloader_creation(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("DataLoader should not be created")

    monkeypatch.setattr(train_module, "get_ts_dls", fail_if_called)
    X_feat = np.ones((4, 10), dtype=np.float32)
    y = np.array(["line", "circle", "other"])
    splits = (np.array([0, 1, 2]), np.array([3]))

    with pytest.raises(ValueError, match="same number of samples"):
        train(X_feat, y, splits, lr=0.001)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("epochs", 0), ("batch_size", -1), ("lr", np.inf)],
)
def test_train_rejects_invalid_hyperparameters(argument, value):
    X_feat = np.ones((4, 10), dtype=np.float32)
    y = np.array(["line", "circle", "other", "line"])
    splits = (np.array([0, 1, 2]), np.array([3]))
    kwargs = {"epochs": 1, "batch_size": 2, "lr": 0.001, argument: value}

    with pytest.raises(ValueError, match=argument):
        train(X_feat, y, splits, **kwargs)


def test_save_artifacts_rejects_unsafe_tag_before_writing(tmp_path):
    with pytest.raises(ValueError, match="path separators"):
        save_artifacts(
            object(),
            object(),
            np.ones((1, 6, 96)),
            output_dir=tmp_path,
            tag="../model",
        )

    assert not list(tmp_path.iterdir())


def test_save_artifacts_persists_full_imu_configuration(tmp_path):
    class FakeFeatureExtractor:
        def state_dict(self):
            return {}

    class FakeLearner:
        def export(self, path):
            Path(path).touch()

    save_artifacts(
        FakeFeatureExtractor(),
        FakeLearner(),
        np.ones((1, 6, 96), dtype=np.float32),
        output_dir=tmp_path,
        tag="example",
        config=DEFAULT_IMU_CONFIG,
    )

    metadata = train_module.torch.load(tmp_path / "input_shape-example.pt")
    assert metadata == {
        "n_channels": 6,
        "n_timesteps": 96,
        "sample_rate_hz": 48.0,
        "window_duration_s": 2.0,
        "channel_names": list(DEFAULT_IMU_CONFIG.channel_names),
    }
