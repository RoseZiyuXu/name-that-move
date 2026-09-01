"""Optional client for the existing remote MiniRocket model server."""

from __future__ import annotations

import pickle
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np

from name_that_move.preprocessing import validate_windows
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.window_buffer import CompletedWindow

RemotePrediction = Prediction


class RemoteInferenceError(RuntimeError):
    """Concise user-facing failure from a remote HTTP inference request."""


class RemoteModelClient:
    """Upload windows to the prototype HTTP inference endpoint."""

    def __init__(
        self,
        url: str,
        *,
        http_timeout_s: float = 2.0,
        session: Any = None,
    ) -> None:
        """Configure the endpoint, timeout, and optional HTTP session."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                "remote_url must be a complete http:// or https:// URL"
            )
        if parsed_url.hostname and parsed_url.hostname.endswith(".example"):
            raise ValueError(
                "remote_url is still a documentation placeholder. Replace "
                "'https://your-model-server.example/process' with the actual "
                "model-server URL."
            )
        if not np.isfinite(http_timeout_s) or http_timeout_s <= 0:
            raise ValueError("http_timeout_s must be positive and finite")
        request_error_types: tuple[type[Exception], ...] = (OSError,)
        if session is None:
            try:
                import requests
            except ImportError as error:
                raise ImportError(
                    "Remote inference requires the 'remote' optional dependencies; "
                    "install with python -m pip install '.[remote]'"
                ) from error
            session = requests.Session()
            request_error_types = (requests.RequestException, OSError)
        else:
            try:
                import requests
            except ImportError:
                pass
            else:
                request_error_types = (requests.RequestException, OSError)
        self.url = url
        self.hostname = parsed_url.hostname
        self.http_timeout_s = float(http_timeout_s)
        self.session = session
        self._request_error_types = request_error_types

    def predict(self, window: CompletedWindow) -> Prediction:
        """Upload one window from an inference worker and parse its response."""
        batch = validate_windows(window.data)
        payload = BytesIO()
        pickle.dump(batch, payload)
        payload.seek(0)
        try:
            response = self.session.post(
                self.url,
                files={"file": ("window.pkl", payload, "application/octet-stream")},
                timeout=self.http_timeout_s,
            )
            response.raise_for_status()
        except self._request_error_types as error:
            raise RemoteInferenceError(
                f"Remote HTTP inference failed for host '{self.hostname}'. "
                "Check the model URL, network connection, server availability, "
                f"and --http-timeout ({self.http_timeout_s:.1f}s)."
            ) from error
        try:
            result = response.json()["result"]
            label = result[0][0]
            confidence = float(result[1][0])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("remote server returned an invalid prediction") from error
        return Prediction(label=str(label), confidence=confidence)
