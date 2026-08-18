"""Optional client for the existing remote MiniRocket model server."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

from minirocket_on_the_fly.preprocessing import validate_windows
from minirocket_on_the_fly.realtime.window_buffer import CompletedWindow


@dataclass(frozen=True)
class RemotePrediction:
    """Label and confidence returned by a remote model server."""

    label: str
    confidence: float


class RemoteModelClient:
    """Upload windows to the prototype HTTP inference endpoint."""

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 5.0,
        session: Any = None,
    ) -> None:
        """Configure the endpoint, timeout, and optional HTTP session."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if session is None:
            try:
                import requests
            except ImportError as error:
                raise ImportError(
                    "Remote inference requires the 'remote' optional dependencies; "
                    "install with python -m pip install '.[remote]'"
                ) from error
            session = requests.Session()
        self.url = url
        self.timeout_s = float(timeout_s)
        self.session = session

    def predict(self, window: CompletedWindow) -> RemotePrediction:
        """Upload one window from an inference worker and parse its response."""
        batch = validate_windows(window.data)
        payload = BytesIO()
        pickle.dump(batch, payload)
        payload.seek(0)
        response = self.session.post(
            self.url,
            files={"file": ("window.pkl", payload, "application/octet-stream")},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        try:
            result = response.json()["result"]
            label = result[0][0]
            confidence = float(result[1][0])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("remote server returned an invalid prediction") from error
        return RemotePrediction(label=str(label), confidence=confidence)

