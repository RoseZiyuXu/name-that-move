"""Command-line entry point for recording labeled OSC IMU windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from name_that_move.preprocessing import IMUWindowConfig
from name_that_move.realtime.osc_receiver import (
    DEFAULT_IMU_ID,
    DEFAULT_OSC_IP,
    DEFAULT_OSC_PORT,
    osc_channel_paths,
)
from name_that_move.realtime.pipeline import RealtimePipeline
from name_that_move.realtime.recorder import AsyncWindowRecorder
from name_that_move.realtime.window_buffer import CompletedWindow


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by ``name-that-move-record``."""
    parser = argparse.ArgumentParser(
        description="Record labeled fixed-rate IMU windows received over OSC."
    )
    parser.add_argument("--label", required=True, help="Motion class, such as line")
    parser.add_argument(
        "--session",
        default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        help="Independent recording-session identifier",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/recordings",
        help="Root directory for generated recordings",
    )
    parser.add_argument(
        "--ip", default=DEFAULT_OSC_IP, help="Local interface to bind"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_OSC_PORT, help="OSC input port"
    )
    parser.add_argument(
        "--imu-id", type=int, default=DEFAULT_IMU_ID, help="Movesense OSC ID"
    )
    parser.add_argument("--sample-rate", type=float, default=48.0)
    parser.add_argument("--window-duration", type=float, default=2.0)
    parser.add_argument(
        "--stale-warning",
        type=float,
        default=0.25,
        help="Warn when any channel has not updated for this many seconds",
    )
    return parser


def main() -> None:
    """Run a recording session until interrupted with Control-C."""
    args = build_parser().parse_args()
    config = IMUWindowConfig(
        sample_rate_hz=args.sample_rate,
        window_duration_s=args.window_duration,
    )
    recorder = AsyncWindowRecorder(
        args.output_dir,
        label=args.label,
        session=args.session,
        imu_id=args.imu_id,
        config=config,
    )

    def report_window(window: CompletedWindow) -> None:
        diagnostics = window.diagnostics
        message = (
            f"Captured {window.data.shape} window | "
            f"OSC messages: {diagnostics.osc_message_count} | "
            f"max channel age: {diagnostics.max_channel_age_s:.3f}s"
        )
        print(message)
        if diagnostics.max_channel_age_s > args.stale_warning:
            print("Warning: at least one channel may be stale")

    pipeline = RealtimePipeline(
        config=config,
        ip=args.ip,
        port=args.port,
        imu_id=args.imu_id,
        recorder=recorder,
        on_window=report_window,
    )

    print(f"Listening on {args.ip}:{args.port} for IMU {args.imu_id}")
    print("Expected OSC addresses:")
    for channel, path in osc_channel_paths(args.imu_id).items():
        print(f"  {channel}: {path}")
    print(f"Saving to: {recorder.recording_dir}")
    print("Waiting until all six channels have arrived; press Control-C to stop.")
    pipeline.run_forever()


if __name__ == "__main__":
    main()
