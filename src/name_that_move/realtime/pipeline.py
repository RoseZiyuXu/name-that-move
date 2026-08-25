"""Orchestration for continuous OSC sampling and downstream workers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Event, Thread
from time import monotonic

from name_that_move.preprocessing import DEFAULT_IMU_CONFIG, IMUWindowConfig
from name_that_move.realtime.inference_worker import InferenceWorker
from name_that_move.realtime.osc_receiver import OSCReceiver
from name_that_move.realtime.recorder import AsyncWindowRecorder
from name_that_move.realtime.window_buffer import (
    CompletedWindow,
    LatestValueWindowBuffer,
)

LOGGER = logging.getLogger(__name__)


class RealtimePipeline:
    """Combine OSC acquisition, fixed-rate sampling, saving, and inference."""

    def __init__(
        self,
        *,
        config: IMUWindowConfig = DEFAULT_IMU_CONFIG,
        ip: str = "0.0.0.0",
        port: int = 10000,
        imu_id: int = 2,
        recorder: AsyncWindowRecorder | None = None,
        inference_worker: InferenceWorker | None = None,
        on_window: Callable[[CompletedWindow], None] | None = None,
    ) -> None:
        """Configure a pipeline without starting its network or worker loop."""
        if recorder is not None and recorder.config != config:
            raise ValueError(
                "Recorder configuration must match pipeline configuration. "
                "Pass the same IMUWindowConfig instance or equivalent values "
                "to both components."
            )
        self.config = config
        self.buffer = LatestValueWindowBuffer(config)
        self.receiver = OSCReceiver(
            self.buffer.update,
            ip=ip,
            port=port,
            imu_id=imu_id,
        )
        self.recorder = recorder
        self.inference_worker = inference_worker
        self.on_window = on_window
        self._stop_event = Event()
        self._sampling_thread: Thread | None = None

    def start(self) -> None:
        """Start OSC reception and fixed-rate latest-value sampling."""
        if self._sampling_thread is not None:
            raise RuntimeError("real-time pipeline is already running")
        self.receiver.start()
        self._stop_event.clear()
        self._sampling_thread = Thread(
            target=self._sampling_loop,
            name="imu-fixed-rate-sampler",
            daemon=True,
        )
        self._sampling_thread.start()

    def stop(self) -> None:
        """Stop acquisition and flush the configured background workers."""
        self._stop_event.set()
        self.receiver.stop()
        if self._sampling_thread is not None:
            self._sampling_thread.join()
            self._sampling_thread = None
        if self.recorder is not None:
            self.recorder.close()
        if self.inference_worker is not None:
            self.inference_worker.close()

    def run_forever(self) -> None:
        """Run until interrupted by the user, then shut down cleanly."""
        try:
            self.start()
            while not self._stop_event.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _sampling_loop(self) -> None:
        period_s = 1.0 / self.config.sample_rate_hz
        next_sample_at = monotonic()
        while not self._stop_event.is_set():
            next_sample_at += period_s
            wait_s = max(0.0, next_sample_at - monotonic())
            if self._stop_event.wait(wait_s):
                return

            sampled_at = monotonic()
            if sampled_at - next_sample_at > period_s:
                next_sample_at = sampled_at
            window = self.buffer.sample(sampled_at=sampled_at)
            if window is None:
                continue

            if self.recorder is not None and not self.recorder.submit(window):
                LOGGER.warning("Recorder queue is full; window was not saved")
            if (
                self.inference_worker is not None
                and not self.inference_worker.submit(window)
            ):
                LOGGER.warning("Inference is busy; window was not classified")
            if self.on_window is not None:
                self.on_window(window)
