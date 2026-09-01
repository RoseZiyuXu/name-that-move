"""OSC input adapter with a configurable prefix and fixed IMU suffixes.

The adapter treats six semantic channel endings—``acc/x``, ``acc/y``,
``acc/z``, ``gyro/x``, ``gyro/y``, and ``gyro/z``—as its acquisition contract.
The leading OSC namespace is configurable for compatibility with different
sender applications.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

from name_that_move._validation import validate_positive_int

OSC_CHANNEL_SUFFIXES = {
    "acc_x": "acc/x",
    "acc_y": "acc/y",
    "acc_z": "acc/z",
    "gyro_x": "gyro/x",
    "gyro_y": "gyro/y",
    "gyro_z": "gyro/z",
}
DEFAULT_OSC_IP = "0.0.0.0"
DEFAULT_OSC_PORT = 10000
DEFAULT_IMU_ID = 1


def validate_osc_prefix(prefix: str) -> str:
    """Validate and remove a trailing slash from an OSC address prefix."""
    if not isinstance(prefix, str):
        raise TypeError("osc_prefix must be a string")
    normalized = prefix.rstrip("/")
    if not normalized or not normalized.startswith("/"):
        raise ValueError("osc_prefix must start with '/' and contain a name")
    if any(character.isspace() for character in normalized):
        raise ValueError("osc_prefix must not contain whitespace")
    return normalized


def osc_channel_paths(
    imu_id: int,
    *,
    prefix: str | None = None,
) -> dict[str, str]:
    """Return the expected OSC address for every six-axis IMU channel.

    When ``prefix`` is omitted, paths use the default ``/m/<imu_id>`` OSC
    convention. A custom prefix changes only that namespace; the six semantic
    channel suffixes remain part of the package data contract.

    Parameters
    ----------
    imu_id:
        Sensor identifier used to construct the default ``/m/<imu_id>``
        prefix. It remains required metadata when a custom prefix is supplied.
    prefix:
        Optional namespace beginning with ``/``, for example
        ``/wearable/right-wrist``. A trailing slash is removed.

    Returns
    -------
    dict
        Package channel names mapped to their complete OSC addresses.
    """
    imu_id = validate_positive_int(imu_id, name="imu_id")
    base_path = f"/m/{imu_id}" if prefix is None else validate_osc_prefix(prefix)
    return {
        channel: f"{base_path}/{suffix}"
        for channel, suffix in OSC_CHANNEL_SUFFIXES.items()
    }


class OSCReceiver:
    """Receive six named IMU values on a background UDP server.

    The default namespace is ``/m/<imu_id>``. ``osc_prefix`` may replace that
    leading namespace, but the receiver still appends and requires the six
    ``acc|gyro`` and ``x|y|z`` suffixes defined by :data:`OSC_CHANNEL_SUFFIXES`.
    """

    def __init__(
        self,
        on_value: Callable[[str, float], None],
        *,
        ip: str = DEFAULT_OSC_IP,
        port: int = DEFAULT_OSC_PORT,
        imu_id: int = DEFAULT_IMU_ID,
        osc_prefix: str | None = None,
    ) -> None:
        """Configure the port, sensor metadata, and optional OSC prefix."""
        if not callable(on_value):
            raise TypeError("on_value must be callable")
        self.on_value = on_value
        self.ip = ip
        self.port = validate_positive_int(port, name="port")
        self.imu_id = validate_positive_int(imu_id, name="imu_id")
        self.osc_prefix = (
            None if osc_prefix is None else validate_osc_prefix(osc_prefix)
        )
        self._server: Any = None
        self._thread: Thread | None = None

    @property
    def channel_paths(self) -> dict[str, str]:
        """Expected OSC addresses keyed by package channel name."""
        return osc_channel_paths(self.imu_id, prefix=self.osc_prefix)

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

    def _handle_message(
        self,
        address: str,
        mapped_channel: Any,
        *args: Any,
    ) -> None:
        del address
        # python-osc passes Dispatcher.map() arguments to handlers as a list.
        # Older versions and simple test doubles may pass the value directly,
        # so accept both forms while still requiring exactly one channel name.
        if isinstance(mapped_channel, (list, tuple)):
            if len(mapped_channel) != 1:
                return
            channel = mapped_channel[0]
        else:
            channel = mapped_channel
        if not isinstance(channel, str):
            return
        if len(args) != 1:
            return
        try:
            value = float(args[0])
            self.on_value(channel, value)
        except (TypeError, ValueError):
            return
