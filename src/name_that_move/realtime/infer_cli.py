"""Command-line entry point for live local or remote movement inference."""

from __future__ import annotations

import argparse

from name_that_move.preprocessing import IMUWindowConfig
from name_that_move.realtime.inference_worker import InferenceWorker
from name_that_move.realtime.osc_receiver import (
    DEFAULT_IMU_ID,
    DEFAULT_OSC_IP,
    DEFAULT_OSC_PORT,
    osc_channel_paths,
)
from name_that_move.realtime.pipeline import RealtimePipeline
from name_that_move.realtime.prediction import Prediction
from name_that_move.realtime.predictor import build_predictor
from name_that_move.realtime.touchdesigner import TouchDesignerClient
from name_that_move.realtime.window_buffer import CompletedWindow


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for live local or remote inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify live OSC IMU windows with either local saved artifacts "
            "or a remote HTTP model server."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model-dir",
        help="Local directory containing MRF, MRL, and input-shape artifacts",
    )
    mode.add_argument("--remote-url", help="Remote HTTP inference endpoint")
    parser.add_argument(
        "--model-tag", default="name_that_move", help="Local artifact tag"
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=2.0,
        help="Maximum wait for each remote HTTP inference request, in seconds",
    )
    parser.add_argument("--ip", default=DEFAULT_OSC_IP, help="Local OSC interface")
    parser.add_argument("--port", type=int, default=DEFAULT_OSC_PORT)
    parser.add_argument("--imu-id", type=int, default=DEFAULT_IMU_ID)
    parser.add_argument("--sample-rate", type=float, default=48.0)
    parser.add_argument("--window-duration", type=float, default=2.0)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=2.0,
        help="Stop if all six OSC channels do not arrive within this many seconds",
    )
    parser.add_argument(
        "--accelerated",
        action="store_true",
        help="Allow local inference to use an available accelerated device",
    )
    parser.add_argument(
        "--touchdesigner-port",
        type=int,
        help="Also send label and confidence to this TouchDesigner OSC port",
    )
    parser.add_argument("--touchdesigner-ip", default="127.0.0.1")
    parser.add_argument("--touchdesigner-path", default="/python")
    return parser


def main() -> None:
    """Validate configuration, then run live inference until Control-C."""
    args = build_parser().parse_args()
    print("\nStarting Name That Move live inference...", flush=True)
    config = IMUWindowConfig(
        sample_rate_hz=args.sample_rate,
        window_duration_s=args.window_duration,
    )
    if args.model_dir is not None:
        print("\nLoading local model...", flush=True)
    else:
        print("\nPreparing remote HTTP model client...", flush=True)
    try:
        predictor = build_predictor(
            model_dir=args.model_dir,
            remote_url=args.remote_url,
            tag=args.model_tag,
            config=config,
            cpu=not args.accelerated,
            http_timeout_s=args.http_timeout,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
    if args.model_dir is not None:
        print("Local model ready.", flush=True)
    else:
        print("Remote HTTP model client ready.", flush=True)
    touchdesigner = None
    if args.touchdesigner_port is not None:
        touchdesigner = TouchDesignerClient(
            ip=args.touchdesigner_ip,
            port=args.touchdesigner_port,
            base_path=args.touchdesigner_path,
        )

    def report_prediction(
        result: Prediction,
        window: CompletedWindow,
    ) -> None:
        print(
            f"Prediction: {result.label} | confidence: "
            f"{result.confidence:.3f}",
            flush=True,
        )
        if touchdesigner is not None:
            touchdesigner.send(result, window)

    def report_error(error: Exception, window: CompletedWindow) -> None:
        del window
        print(f"Inference error: {error}", flush=True)

    worker = InferenceWorker(
        predictor.predict,
        on_result=report_prediction,
        on_error=report_error,
    )
    pipeline = RealtimePipeline(
        config=config,
        ip=args.ip,
        port=args.port,
        imu_id=args.imu_id,
        inference_worker=worker,
        startup_timeout_s=args.startup_timeout,
    )

    mode_name = "local" if args.model_dir is not None else "remote"
    print("\nStarting OSC receiver...", flush=True)
    print("\nLive inference configuration")
    print(f"  Mode: {mode_name}")
    print(f"  Listening on: {args.ip}:{args.port}")
    print(f"  IMU ID: {args.imu_id}")
    print("\nExpected OSC addresses")
    for channel, path in osc_channel_paths(args.imu_id).items():
        print(f"  {channel}: {path}")
    print(
        "\nWaiting until all six channels have arrived. "
        "Press Control-C to stop.",
        flush=True,
    )
    try:
        pipeline.run_forever()
    except TimeoutError as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
