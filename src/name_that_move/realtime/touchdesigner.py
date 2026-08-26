"""Optional OSC output adapter for TouchDesigner."""

from __future__ import annotations

from typing import Any

from name_that_move._validation import validate_positive_int
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.window_buffer import CompletedWindow


class TouchDesignerClient:
    """Send prediction labels and confidence values to TouchDesigner."""

    def __init__(
        self,
        *,
        ip: str = "127.0.0.1",
        port: int = 8000,
        base_path: str = "/python",
        client: Any = None,
    ) -> None:
        """Configure the target OSC address and optional injected client."""
        if not isinstance(base_path, str) or not base_path.startswith("/"):
            raise ValueError("base_path must be an OSC path beginning with '/'")
        if client is None:
            try:
                from pythonosc.udp_client import SimpleUDPClient
            except ImportError as error:
                raise ImportError(
                    "TouchDesigner output requires the 'realtime' optional "
                    "dependencies; install with python -m pip install '.[realtime]'"
                ) from error
            client = SimpleUDPClient(ip, validate_positive_int(port, name="port"))
        self.base_path = base_path.rstrip("/")
        self.client = client

    def send(
        self,
        prediction: Prediction,
        window: CompletedWindow | None = None,
    ) -> None:
        """Send one prediction; the optional window supports worker callbacks."""
        del window
        self.client.send_message(f"{self.base_path}/label", prediction.label)
        self.client.send_message(
            f"{self.base_path}/confidence", float(prediction.confidence)
        )
