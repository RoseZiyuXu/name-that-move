import pickle

import numpy as np

from minirocket_on_the_fly.infer import load_segment, load_segments_batch


def test_load_segment_accepts_wrapped_array(tmp_path):
    expected = np.ones((6, 48), dtype=np.float32)
    path = tmp_path / "segment.pkl"
    with path.open("wb") as file:
        pickle.dump([expected], file)

    np.testing.assert_array_equal(load_segment(path), expected)


def test_load_segments_batch_adds_batch_dimension(tmp_path):
    expected = np.ones((6, 48), dtype=np.float32)
    path = tmp_path / "segment.pkl"
    with path.open("wb") as file:
        pickle.dump(expected, file)

    result = load_segments_batch(path)

    assert result.shape == (1, 6, 48)
