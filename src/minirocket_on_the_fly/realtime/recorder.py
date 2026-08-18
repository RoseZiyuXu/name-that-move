"""Asynchronous persistence for real-time IMU windows."""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread

from minirocket_on_the_fly._validation import validate_path, validate_positive_int
from minirocket_on_the_fly.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    validate_windows,
)
from minirocket_on_the_fly.realtime.window_buffer import CompletedWindow

_STOP = object()
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_component(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must use only letters, numbers, periods, underscores, "
            "or hyphens"
        )
    return value


class AsyncWindowRecorder:
    """Save captured windows without blocking the sampling thread."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        label: str,
        session: str,
        imu_id: int,
        config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
        max_queue_size: int = 16,
    ) -> None:
        """Create and start a background file-writing worker."""
        self.output_dir = Path(validate_path(output_dir, name="output_dir"))
        self.label = _validate_component(label, name="label")
        self.session = _validate_component(session, name="session")
        self.imu_id = validate_positive_int(imu_id, name="imu_id")
        self.config = config
        self._queue: Queue[CompletedWindow | object] = Queue(
            validate_positive_int(max_queue_size, name="max_queue_size")
        )
        self._error: Exception | None = None
        self._closed = False
        self._thread = Thread(target=self._run, name="imu-window-recorder", daemon=True)
        self._thread.start()

    @property
    def recording_dir(self) -> Path:
        """Directory containing this label and session's recorded windows."""
        return self.output_dir / self.label / self.session

    def submit(self, window: CompletedWindow) -> bool:
        """Queue a window for saving and return immediately."""
        if self._closed:
            raise RuntimeError("recorder is closed")
        if self._error is not None:
            raise RuntimeError("recorder worker failed") from self._error
        validate_windows(window.data, config=self.config)
        try:
            self._queue.put_nowait(window)
        except Full:
            return False
        return True

    def close(self) -> None:
        """Flush pending windows and stop the file-writing worker."""
        if self._closed:
            return
        self._closed = True
        if self._thread.is_alive():
            self._queue.put(_STOP)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("recorder worker failed") from self._error

    def _run(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    if item is _STOP:
                        return
                    self._write(item)
                finally:
                    self._queue.task_done()
        # Worker boundary: preserve an unexpected file-writing failure so the
        # caller receives it on submit or close instead of losing it silently.
        except Exception as error:  # noqa: BLE001
            self._error = error

    def _write(self, window: CompletedWindow | object) -> None:
        if not isinstance(window, CompletedWindow):
            raise TypeError("recorder queue contained an invalid window")
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        captured_at = datetime.now(timezone.utc)
        timestamp = captured_at.strftime("%Y%m%dT%H%M%S_%fZ")
        stem = f"imu{self.imu_id}_{timestamp}"
        data_path = self.recording_dir / f"{stem}.pkl"
        metadata_path = self.recording_dir / f"{stem}.json"

        with data_path.open("wb") as file:
            pickle.dump(window.data, file)

        metadata = {
            "captured_at_utc": captured_at.isoformat(),
            "imu_id": self.imu_id,
            "label": self.label,
            "session": self.session,
            "sample_rate_hz": self.config.sample_rate_hz,
            "window_duration_s": self.config.window_duration_s,
            "channel_names": list(self.config.channel_names),
            "shape": list(window.data.shape),
            "osc_message_count": window.diagnostics.osc_message_count,
            "max_channel_age_s": window.diagnostics.max_channel_age_s,
        }
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
