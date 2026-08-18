import pickle

import numpy as np
import pytest

from minirocket_on_the_fly import infer
from minirocket_on_the_fly.infer import (
    load_model,
    load_segment,
    load_segments_batch,
    predict,
)


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
