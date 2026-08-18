"""OSC input adapter for Movesense IMU channel messages."""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

from minirocket_on_the_fly._validation import validate_positive_int

OSC_CHANNEL_SUFFIXES = {
    "acc_x": "acc/x",
    "acc_y": "acc/y",
    "acc_z": "acc/z",
    "gyro_x": "gyro/x",
    "gyro_y": "gyro/y",
    "gyro_z": "gyro/z",
}


def osc_channel_paths(imu_id: int) -> dict[str, str]:
    """Return the expected OSC address for every six-axis IMU channel."""
    imu_id = validate_positive_int(imu_id, name="imu_id")
    return {
        channel: f"/m/{imu_id}/{suffix}"
        for channel, suffix in OSC_CHANNEL_SUFFIXES.items()
    }


class OSCReceiver:
    """Receive named OSC values on a background UDP server."""

    def __init__(
        self,
        on_value: Callable[[str, float], None],
        *,
        ip: str = "0.0.0.0",
        port: int = 10000,
        imu_id: int = 2,
    ) -> None:
        """Configure an OSC receiver without opening the network port yet."""
        if not callable(on_value):
            raise TypeError("on_value must be callable")
        self.on_value = on_value
        self.ip = ip
        self.port = validate_positive_int(port, name="port")
        self.imu_id = validate_positive_int(imu_id, name="imu_id")
        self._server: Any = None
        self._thread: Thread | None = None

    @property
    def channel_paths(self) -> dict[str, str]:
        """Expected OSC addresses keyed by package channel name."""
        return osc_channel_paths(self.imu_id)

    def start(self) -> None:
        """Start receiving OSC messages in a background thread."""
        if self._thread is not None:
            raise RuntimeError("OSC receiver is already running")
        try:
            from pythonosc.dispatcher import Dispatcher
            from pythonosc.osc_server import ThreadingOSCUDPServer
        except ImportError as error:
            raise ImportError(
                "OSC recording requires the 'realtime' optional dependencies; "
                "install with python -m pip install '.[realtime]'"
            ) from error

        dispatcher = Dispatcher()
        for channel, path in self.channel_paths.items():
            dispatcher.map(path, self._handle_message, channel)

        self._server = ThreadingOSCUDPServer((self.ip, self.port), dispatcher)
        self._thread = Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="imu-osc-receiver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop receiving messages and release the UDP port."""
        if self._server is None or self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self._server = None
        self._thread = None

    def _handle_message(self, address: str, channel: str, *args: Any) -> None:
        if len(args) != 1:
            return
        try:
            value = float(args[0])
            self.on_value(channel, value)
        except (TypeError, ValueError):
            return

