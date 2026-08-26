"""Command-line entry point for offline session evaluation."""

from __future__ import annotations

import argparse

from name_that_move.offline.evaluation import evaluate_session
from name_that_move.preprocessing import IMUWindowConfig
from name_that_move.realtime.remote_client import RemoteInferenceError


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for saved-model evaluation on one session folder."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on a folder of recorded IMU windows."
    )
    parser.add_argument("--session-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-dir", help="Local saved-model artifact directory")
    mode.add_argument("--remote-url", help="Remote HTTP inference endpoint")
    parser.add_argument("--model-tag", default="name_that_move")
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=2.0,
        help="Maximum wait for each remote HTTP inference request, in seconds",
    )
    parser.add_argument(
        "--expected-label",
        help="Ground-truth class shared by the session, such as circle",
    )
    parser.add_argument("--sample-rate", type=float, default=48.0)
    parser.add_argument("--window-duration", type=float, default=2.0)
    parser.add_argument(
        "--accelerated",
        action="store_true",
        help="Allow inference to use an available accelerated device",
    )
    return parser


def main() -> None:
    """Evaluate the selected session and print a compact summary."""
    args = build_parser().parse_args()
    config = IMUWindowConfig(
        sample_rate_hz=args.sample_rate,
        window_duration_s=args.window_duration,
    )
    try:
        result = evaluate_session(
            args.session_dir,
            args.model_dir,
            remote_url=args.remote_url,
            tag=args.model_tag,
            expected_label=args.expected_label,
            config=config,
            cpu=not args.accelerated,
            http_timeout_s=args.http_timeout,
        )
    except (FileNotFoundError, RemoteInferenceError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
    counts = ", ".join(
        f"'{label}'={count}" for label, count in sorted(result.label_counts.items())
    )
    print(f"Session: {result.session_dir}")
    print(f"Windows: {result.n_windows}")
    print(f"Predicted labels: {counts}")
    print(f"Mean confidence: {result.mean_confidence:.3f}")
    if result.accuracy is not None:
        correct = round(result.accuracy * result.n_windows)
        print(
            f"Accuracy for expected label '{result.expected_label}': "
            f"{correct}/{result.n_windows} ({result.accuracy:.1%})"
        )


if __name__ == "__main__":
    main()
