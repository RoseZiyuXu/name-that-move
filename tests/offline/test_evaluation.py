import pickle

import numpy as np
import pytest

from name_that_move.offline import evaluation


def test_evaluate_session_summarizes_predictions(tmp_path, monkeypatch):
    session_dir = tmp_path / "circle4"
    session_dir.mkdir()
    for name in ("window_02.pkl", "window_01.pkl", "window_03.pkl"):
        (session_dir / name).touch()

    monkeypatch.setattr(
        evaluation,
        "load_segment",
        lambda path: np.ones((6, 96), dtype=np.float32),
    )
    monkeypatch.setattr(evaluation, "load_model", lambda *args, **kwargs: (1, 2))
    monkeypatch.setattr(
        evaluation,
        "predict",
        lambda *args, **kwargs: (
            np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]),
            ["circle", "circle", "still"],
        ),
    )

    result = evaluation.evaluate_session(
        session_dir,
        tmp_path / "models",
        expected_label="circle",
    )

    assert result.n_windows == 3
    assert result.label_counts == {"circle": 2, "still": 1}
    assert result.mean_confidence == pytest.approx(0.8)
    assert result.accuracy == pytest.approx(2 / 3)
    assert [path.name for path in result.file_paths] == [
        "window_01.pkl",
        "window_02.pkl",
        "window_03.pkl",
    ]


def test_evaluate_session_requires_pickle_windows(tmp_path):
    session_dir = tmp_path / "empty_session"
    session_dir.mkdir()

    with pytest.raises(ValueError, match="No .pkl windows"):
        evaluation.evaluate_session(session_dir, tmp_path / "models")


def test_evaluate_session_uses_remote_http_model(tmp_path):
    session_dir = tmp_path / "triangle4"
    session_dir.mkdir()
    for index in range(2):
        path = session_dir / f"window_{index}.pkl"
        with path.open("wb") as file:
            pickle.dump(np.ones((6, 96), dtype=np.float32), file)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": [["triangle"], [0.8]]}

    class FakeSession:
        def __init__(self):
            self.requests = 0

        def post(self, url, *, files, timeout):
            assert url == "https://model.test/process"
            assert timeout == 2.0
            assert pickle.load(files["file"][1]).shape == (1, 6, 96)
            self.requests += 1
            return FakeResponse()

    session = FakeSession()
    result = evaluation.evaluate_session(
        session_dir,
        remote_url="https://model.test/process",
        expected_label="triangle",
        http_timeout_s=2.0,
        remote_session=session,
    )

    assert session.requests == 2
    assert result.label_counts == {"triangle": 2}
    assert result.accuracy == 1.0


def test_evaluate_session_rejects_ambiguous_inference_mode(tmp_path):
    session_dir = tmp_path / "circle4"
    session_dir.mkdir()
    with (session_dir / "window.pkl").open("wb") as file:
        pickle.dump(np.ones((6, 96), dtype=np.float32), file)

    with pytest.raises(ValueError, match="cannot be used together"):
        evaluation.evaluate_session(
            session_dir,
            tmp_path / "models",
            remote_url="https://model.example/process",
        )
