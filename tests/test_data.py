import pickle

import numpy as np
import pytest

from minirocket_on_the_fly import data


def test_load_segments_accepts_bare_and_wrapped_arrays(tmp_path):
    class_dir = tmp_path / "class_zero"
    class_dir.mkdir()
    expected = np.ones((6, 48), dtype=np.float32)

    with (class_dir / "bare.pkl").open("wb") as file:
        pickle.dump(expected, file)
    with (class_dir / "wrapped.pkl").open("wb") as file:
        pickle.dump([expected], file)

    X, y = data.load_segments(tmp_path, file_names=["class_zero"])

    assert X.shape == (2, 6, 48)
    assert y.tolist() == ["0", "0"]


def test_make_dataset_augments_training_samples_only(monkeypatch):
    X_orig = np.arange(10 * 2 * 3).reshape(10, 2, 3)
    y_orig = np.arange(10).astype(str)

    monkeypatch.setattr(
        data,
        "load_segments",
        lambda base_path, file_names: (X_orig, y_orig),
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
