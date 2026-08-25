import pickle

import numpy as np
import pytest

from name_that_move.offline import data
from name_that_move.preprocessing import IMUWindowConfig


def test_load_segments_accepts_bare_and_wrapped_arrays(tmp_path):
    class_dir = tmp_path / "class_zero"
    class_dir.mkdir()
    expected = np.ones((6, 96), dtype=np.float32)

    with (class_dir / "bare.pkl").open("wb") as file:
        pickle.dump(expected, file)
    with (class_dir / "wrapped.pkl").open("wb") as file:
        pickle.dump([expected], file)

    X, y = data.load_segments(tmp_path, file_names=["class_zero"])

    assert X.shape == (2, 6, 96)
    assert y.tolist() == ["0", "0"]


def test_make_dataset_augments_training_samples_only(monkeypatch):
    X_orig = np.arange(10 * 2 * 3).reshape(10, 2, 3)
    y_orig = np.arange(10).astype(str)

    monkeypatch.setattr(
        data,
        "load_segments",
        lambda base_path, file_names, **kwargs: (X_orig, y_orig),
    )

    def fake_augment(X, y, **kwargs):
        return X + 1000, y.copy()

    monkeypatch.setattr(data, "augment_segments", fake_augment)

    X, y, (train_indices, val_indices) = data.make_dataset(
        "unused",
        n_aug=1,
        val_fraction=0.2,
        random_seed=42,
    )

    assert X.dtype == np.float32
    assert len(y) == 18
    assert len(train_indices) == 16
    assert len(val_indices) == 2
    assert np.all(val_indices < len(y_orig))
    assert set(train_indices).isdisjoint(val_indices)


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.1])
def test_make_dataset_rejects_invalid_validation_fraction(val_fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        data.make_dataset("unused", val_fraction=val_fraction)


def test_load_segments_reports_config_mismatch_with_file_path(tmp_path):
    class_dir = tmp_path / "line"
    class_dir.mkdir()
    path = class_dir / "wrong_size.pkl"
    with path.open("wb") as file:
        pickle.dump(np.ones((6, 95), dtype=np.float32), file)

    config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
    with pytest.raises(ValueError, match=r"wrong_size\.pkl.*\(6, 96\).+\(6, 95\)"):
        data.load_segments(tmp_path, file_names=["line"], config=config)


def test_make_session_dataset_groups_classes_and_holds_out_folders(tmp_path):
    class_folders = {"still": ("still1", "still2"), "circle": ("circle1", "circle2")}
    for folder_number, folder in enumerate(
        ("still1", "still2", "circle1", "circle2")
    ):
        folder_path = tmp_path / folder
        folder_path.mkdir()
        for window_number in range(2):
            window = np.full((1, 6, 96), folder_number + window_number / 10)
            with (folder_path / f"window_{window_number}.pkl").open("wb") as file:
                pickle.dump(window, file)

    X, y, (train_indices, val_indices) = data.make_session_dataset(
        tmp_path,
        class_folders,
        validation_folders={"still2", "circle2"},
    )

    assert X.shape == (8, 6, 96)
    assert X.dtype == np.float32
    assert y[train_indices].tolist() == ["still", "still", "circle", "circle"]
    assert y[val_indices].tolist() == ["still", "still", "circle", "circle"]
    assert set(train_indices).isdisjoint(val_indices)


def test_make_session_dataset_augments_training_folders_only(tmp_path, monkeypatch):
    for folder in ("still1", "still2"):
        folder_path = tmp_path / folder
        folder_path.mkdir()
        with (folder_path / "window.pkl").open("wb") as file:
            pickle.dump(np.ones((6, 96)), file)

    monkeypatch.setattr(
        data,
        "augment_segments",
        lambda X, y, **kwargs: (X + 10, y.copy()),
    )
    X, y, (train_indices, val_indices) = data.make_session_dataset(
        tmp_path,
        {"still": ("still1", "still2")},
        validation_folders={"still2"},
        n_aug=1,
    )

    assert X.shape == (3, 6, 96)
    assert y.tolist() == ["still", "still", "still"]
    assert train_indices.tolist() == [0, 2]
    assert val_indices.tolist() == [1]


def test_make_session_dataset_rejects_reused_folder():
    with pytest.raises(ValueError, match="assigned to both"):
        data.make_session_dataset(
            "unused",
            {"still": ("session1", "session2"), "circle": ("session1", "session3")},
            validation_folders={"session2", "session3"},
        )


def test_make_session_dataset_requires_train_and_validation_per_class():
    with pytest.raises(ValueError, match="training folder and one validation folder"):
        data.make_session_dataset(
            "unused",
            {"still": ("still1", "still2"), "circle": ("circle1", "circle2")},
            validation_folders={"still2"},
        )
