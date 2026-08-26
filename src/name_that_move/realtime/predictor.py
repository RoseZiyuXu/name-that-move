"""Fail-fast selection between local and remote real-time inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from name_that_move.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig
from name_that_move.realtime.local_model import LocalModelPredictor
from name_that_move.realtime.remote_client import RemoteModelClient


def build_predictor(
    *,
    model_dir: str | Path | None = None,
    remote_url: str | None = None,
    tag: str = "name_that_move",
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
    cpu: bool = True,
    http_timeout_s: float = 2.0,
    remote_session: Any = None,
) -> LocalModelPredictor | RemoteModelClient:
    """Build exactly one local or remote predictor.

    Parameters
    ----------
    model_dir:
        Directory containing local saved-model artifacts.
    remote_url:
        HTTP inference endpoint used instead of a local model.
    tag:
        Artifact tag for local inference.
    config:
        Runtime IMU contract checked against local model metadata.
    cpu:
        Whether local artifacts should load on CPU.
    http_timeout_s:
        HTTP timeout used only for remote inference.
    remote_session:
        Optional injected HTTP session, primarily for testing.

    Returns
    -------
    LocalModelPredictor or RemoteModelClient
        Predictor exposing ``predict(CompletedWindow) -> Prediction``.

    """
    if model_dir is None and remote_url is None:
        raise ValueError(
            "Choose an inference mode: provide model_dir for local inference "
            "or remote_url for HTTP inference."
        )
    if model_dir is not None and remote_url is not None:
        raise ValueError(
            "Choose only one inference mode: model_dir and remote_url cannot "
            "be used together."
        )
    if model_dir is not None:
        return LocalModelPredictor(
            model_dir,
            tag=tag,
            config=config,
            cpu=cpu,
        )
    assert remote_url is not None
    return RemoteModelClient(
        remote_url,
        http_timeout_s=http_timeout_s,
        session=remote_session,
    )
