import numpy as np
import pytest

from name_that_move.preprocessing import DEFAULT_IMU_CONFIG
from name_that_move.realtime import local_model
from name_that_move.realtime.infer_cli import build_parser as build_live_parser
from name_that_move.realtime.local_model import LocalModelPredictor
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.predictor import build_predictor
from name_that_move.realtime.remote_client import RemoteModelClient
from name_that_move.realtime.window_buffer import (
    CompletedWindow,
    WindowDiagnostics,
)


def make_window():
    return CompletedWindow(
        data=np.ones((6, 96), dtype=np.float32),
        diagnostics=WindowDiagnostics(0.0, 2.0, 576, 0.01),
    )


def test_prediction_validates_confidence():
    assert Prediction("circle", 0.9).label == "circle"

    with pytest.raises(ValueError, match="between 0 and 1"):
        Prediction("circle", 1.1)


def test_local_predictor_returns_shared_prediction(monkeypatch):
    def fake_predict(X, feature_extractor, learner, *, config):
        assert X.shape == (6, 96)
        assert feature_extractor is sentinel_feature_extractor
        assert learner is sentinel_learner
        assert config == DEFAULT_IMU_CONFIG
        return np.array([[0.05, 0.90, 0.05]]), ["triangle"]

    sentinel_feature_extractor = object()
    sentinel_learner = object()
    monkeypatch.setattr(
        local_model,
        "load_model",
        lambda *args, **kwargs: (sentinel_feature_extractor, sentinel_learner),
    )
    monkeypatch.setattr(local_model, "predict", fake_predict)
    predictor = LocalModelPredictor("unused")

    assert predictor.predict(make_window()) == Prediction("triangle", 0.9)


def test_predictor_selection_fails_when_mode_is_ambiguous():
    with pytest.raises(ValueError, match="provide model_dir"):
        build_predictor()

    with pytest.raises(ValueError, match="cannot be used together"):
        build_predictor(model_dir="models", remote_url="https://example.test")


def test_predictor_selection_builds_remote_client():
    session = object()
    predictor = build_predictor(
        remote_url="https://example.test",
        http_timeout_s=2.0,
        remote_session=session,
    )

    assert isinstance(predictor, RemoteModelClient)
    assert predictor.session is session
    assert predictor.http_timeout_s == 2.0


def test_live_cli_defaults_to_two_second_timeouts():
    args = build_live_parser().parse_args(
        ["--remote-url", "https://model.test/process"]
    )

    assert args.http_timeout == 2.0
    assert args.startup_timeout == 2.0


def test_live_cli_accepts_custom_osc_prefix():
    args = build_live_parser().parse_args(
        [
            "--remote-url",
            "https://model.test/process",
            "--osc-prefix",
            "/wearable/right-wrist",
        ]
    )

    assert args.osc_prefix == "/wearable/right-wrist"
