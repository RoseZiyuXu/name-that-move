"""Asynchronous persistence for real-time IMU windows."""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread

from name_that_move._validation import validate_path, validate_positive_int
from name_that_move.preprocessing import (
    DEFAULT_IMU_CONFIG,
    IMUWindowConfig,
    validate_windows,
)
from name_that_move.realtime.window_buffer import CompletedWindow

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
        self._submitted_count = 0
        self._thread = Thread(target=self._run, name="imu-window-recorder", daemon=True)
        self._thread.start()

    @property
    def recording_dir(self) -> Path:
        """Directory containing this label and session's recorded windows."""
        return self.output_dir / self.label / self.session

    @property
    def submitted_count(self) -> int:
        """Number of windows accepted for saving in this session."""
        return self._submitted_count

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
        self._submitted_count += 1
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
        captured_at_local = captured_at.astimezone()
        local_timestamp = (
            f"{captured_at_local:%Y%m%d_%H%M%S}_"
            f"{captured_at_local.microsecond // 10_000:02d}"
        )
        base_stem = f"imu{self.imu_id}_{local_timestamp}"
        stem = base_stem
        duplicate_number = 2
        while (self.recording_dir / f"{stem}.pkl").exists() or (
            self.recording_dir / f"{stem}.json"
        ).exists():
            stem = f"{base_stem}_{duplicate_number:02d}"
            duplicate_number += 1
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
