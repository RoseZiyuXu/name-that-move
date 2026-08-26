"""Adapter for real-time inference with saved local model artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from name_that_move.infer import load_model, predict
from name_that_move.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.window_buffer import CompletedWindow


class LocalModelPredictor:
    """Load one saved model and classify completed IMU windows locally."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        tag: str = "name_that_move",
        config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
        cpu: bool = True,
    ) -> None:
        """Load and validate model artifacts once, before live sampling starts."""
        feature_extractor, learner = load_model(
            model_dir,
            tag,
            expected_config=config,
            cpu=cpu,
        )
        self.config = config
        self.feature_extractor = feature_extractor
        self.learner = learner

    def predict(self, window: CompletedWindow) -> Prediction:
        """Classify one completed window and return a shared result object."""
        probabilities, labels = predict(
            window.data,
            self.feature_extractor,
            self.learner,
            config=self.config,
        )
        if probabilities.shape[0] != 1 or len(labels) != 1:
            raise ValueError("local predictor expected exactly one prediction")
        return Prediction(
            label=str(labels[0]),
            confidence=float(np.max(probabilities[0])),
        )
