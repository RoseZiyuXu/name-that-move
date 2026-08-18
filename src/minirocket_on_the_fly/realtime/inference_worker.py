"""Non-blocking inference execution for real-time windows."""

from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any

from minirocket_on_the_fly._validation import validate_positive_int
from minirocket_on_the_fly.realtime.window_buffer import CompletedWindow

_STOP = object()


class InferenceWorker:
    """Run model or remote inference outside the sampling thread."""

    def __init__(
        self,
        predict_window: Callable[[CompletedWindow], Any],
        *,
        on_result: Callable[[Any, CompletedWindow], None] | None = None,
        on_error: Callable[[BaseException, CompletedWindow], None] | None = None,
        max_queue_size: int = 1,
    ) -> None:
        """Create and start a bounded background inference worker."""
        if not callable(predict_window):
            raise TypeError("predict_window must be callable")
        self._predict_window = predict_window
        self._on_result = on_result
        self._on_error = on_error
        self._queue: Queue[CompletedWindow | object] = Queue(
            validate_positive_int(max_queue_size, name="max_queue_size")
        )
        self._closed = False
        self._thread = Thread(target=self._run, name="imu-inference", daemon=True)
        self._thread.start()

    def submit(self, window: CompletedWindow) -> bool:
        """Queue a window without waiting; return false when inference is busy."""
        if self._closed:
            raise RuntimeError("inference worker is closed")
        try:
            self._queue.put_nowait(window)
        except Full:
            return False
        return True

    def close(self) -> None:
        """Finish pending inference and stop the worker."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if item is _STOP:
                    return
                if not isinstance(item, CompletedWindow):
                    raise TypeError("inference queue contained an invalid window")
                try:
                    result = self._predict_window(item)
                except BaseException as error:
                    if self._on_error is not None:
                        self._on_error(error, item)
                else:
                    if self._on_result is not None:
                        self._on_result(result, item)
            finally:
                self._queue.task_done()

