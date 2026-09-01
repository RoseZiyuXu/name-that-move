"""Shared prediction result for local and remote real-time inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """One predicted movement label and its confidence score."""

    label: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate result fields at the local/remote adapter boundary."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("prediction label must be a non-empty string")
        if not np.isfinite(self.confidence):
            raise ValueError("prediction confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be between 0 and 1")
