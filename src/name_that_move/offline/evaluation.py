"""Evaluate a saved model on a later, offline recording session."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from name_that_move.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig
from name_that_move.realtime.remote_client import RemoteModelClient
from name_that_move.realtime.window_buffer import (
    CompletedWindow,
    WindowDiagnostics,
)
from name_that_move.window_io import load_segment


@dataclass(frozen=True)
class SessionEvaluation:
    """Predictions and summary metrics for one offline session folder."""

    session_dir: Path
    file_paths: tuple[Path, ...]
    predicted_labels: tuple[str, ...]
    confidences: np.ndarray
    expected_label: str | None = None

    @property
    def n_windows(self) -> int:
        """Return the number of evaluated windows."""
        return len(self.predicted_labels)

    @property
    def label_counts(self) -> dict[str, int]:
        """Return the number of predictions assigned to each class."""
        return dict(Counter(self.predicted_labels))

    @property
    def mean_confidence(self) -> float:
        """Return mean maximum-class probability across the session."""
        return float(np.mean(self.confidences))

    @property
    def accuracy(self) -> float | None:
        """Return accuracy when one expected label was supplied."""
        if self.expected_label is None:
            return None
        return float(
            np.mean(np.asarray(self.predicted_labels) == self.expected_label)
        )


def evaluate_session(
    session_dir: str | Path,
    model_dir: str | Path | None = None,
    *,
    remote_url: str | None = None,
    tag: str = "name_that_move",
    expected_label: str | None = None,
    config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
    cpu: bool = True,
    http_timeout_s: float = 2.0,
    remote_session: object | None = None,
) -> SessionEvaluation:
    """Run batch inference over every ``.pkl`` window in one session.

    Use this for data recorded after model training, such as a held-out
    Session 4. The session remains offline: no OSC stream or real-time timing
    is required.

    Parameters
    ----------
    session_dir:
        Flat directory containing the session's ``.pkl`` windows.
    model_dir:
        Directory containing saved MiniRocket model artifacts for local
        inference. Provide exactly one of ``model_dir`` and ``remote_url``.
    remote_url:
        HTTP inference endpoint used instead of local model artifacts.
    tag:
        Tag shared by the three saved artifact filenames.
    expected_label:
        Optional ground-truth class shared by all windows in this directory.
        When supplied, the result includes accuracy.
    config:
        IMU data contract used for recording, training, and inference.
    cpu:
        Whether to load model artifacts on CPU.
    http_timeout_s:
        Maximum wait for each remote HTTP inference request.
    remote_session:
        Optional injected HTTP session, primarily for testing.

    Returns
    -------
    SessionEvaluation
        Per-window predictions plus counts, confidence, and optional accuracy.

    """
    session_path = Path(session_dir)
    if not session_path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_path}")
    file_paths = tuple(
        sorted(path for path in session_path.iterdir() if path.suffix == ".pkl")
    )
    if not file_paths:
        raise ValueError(f"No .pkl windows found in session: {session_path}")
    if expected_label is not None:
        if not isinstance(expected_label, str) or not expected_label.strip():
            raise ValueError("expected_label must be a non-empty string")

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
    remote_client = None
    if remote_url is not None:
        remote_client = RemoteModelClient(
            remote_url,
            http_timeout_s=http_timeout_s,
            session=remote_session,
        )

    windows = np.stack([load_segment(path) for path in file_paths])
    if model_dir is not None:
        probabilities, predicted_labels = _evaluate_local(
            windows, model_dir, tag=tag, config=config, cpu=cpu
        )
        confidences = np.max(probabilities, axis=1)
    else:
        assert remote_client is not None
        diagnostics = WindowDiagnostics(0.0, 0.0, 0, 0.0)
        remote_predictions = [
            remote_client.predict(CompletedWindow(window, diagnostics))
            for window in windows
        ]
        predicted_labels = [result.label for result in remote_predictions]
        confidences = np.asarray(
            [result.confidence for result in remote_predictions], dtype=float
        )
    return SessionEvaluation(
        session_dir=session_path,
        file_paths=file_paths,
        predicted_labels=tuple(str(label) for label in predicted_labels),
        confidences=confidences,
        expected_label=expected_label,
    )


def _evaluate_local(
    windows: np.ndarray,
    model_dir: str | Path,
    *,
    tag: str,
    config: IMUWindowConfig,
    cpu: bool,
) -> tuple[np.ndarray, list]:
    """Load local ML dependencies only when local inference is requested."""
    from name_that_move.infer import load_model, predict

    feature_extractor, learner = load_model(
        model_dir,
        tag,
        expected_config=config,
        cpu=cpu,
    )
    return predict(
        windows,
        feature_extractor,
        learner,
        config=config,
    )
