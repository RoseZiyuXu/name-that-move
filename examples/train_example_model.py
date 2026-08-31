"""Train, save, reload, and evaluate the three-class example IMU model.

The dataset was captured in three separate recording rounds. The first round
produced ``still1``, ``triangle1``, and ``circle1``; the second produced the
folders ending in ``2``; and the third produced the folders ending in ``3``.
The suffix therefore identifies a recording session, not a motion class.

This example trains on Sessions 1 and 2 and holds out all of Session 3. That
cross-session split tests transfer to a later recording and avoids mixing
closely related windows from one session across training and validation. It is
a package workflow demonstration, not a cross-performer or cross-device
benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from name_that_move import (
    IMUWindowConfig,
    extract_features,
    load_model,
    make_session_dataset,
    predict,
    save_artifacts,
    train,
)

CLASS_FOLDERS = {
    "still": ("still1", "still2", "still3"),
    "triangle": ("triangle1", "triangle2", "triangle3"),
    "circle": ("circle1", "circle2", "circle3"),
}
VALIDATION_FOLDERS = {"still3", "triangle3", "circle3"}
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "still_triangle_circle"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the example training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Directory containing still1..3, triangle1..3, and circle1..3 "
            "(default: the repository's public example dataset)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/example_data_smoke"),
        help="Directory for the three saved model artifacts",
    )
    parser.add_argument("--tag", default="still_triangle_circle_v0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run the reproducible session-level training and inference smoke test."""
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)

    X, y, splits = make_session_dataset(
        args.dataset,
        CLASS_FOLDERS,
        validation_folders=VALIDATION_FOLDERS,
        config=config,
    )
    X_features, feature_extractor = extract_features(X, splits, config=config)
    learner = train(
        X_features,
        y,
        splits,
        epochs=args.epochs,
        batch_size=32,
        lr=args.learning_rate,
        show_graph=False,
    )
    save_artifacts(
        feature_extractor,
        learner,
        X,
        output_dir=args.output_dir,
        tag=args.tag,
        config=config,
    )

    loaded_extractor, loaded_learner = load_model(
        args.output_dir,
        tag=args.tag,
        expected_config=config,
    )
    validation_indices = splits[1]
    _, predictions = predict(
        X[validation_indices],
        loaded_extractor,
        loaded_learner,
        config=config,
    )
    truth = y[validation_indices]
    predictions = np.asarray(predictions, dtype=str)
    correct = int(np.sum(predictions == truth))
    print(
        f"Reloaded-model validation accuracy: {correct}/{len(truth)} "
        f"({correct / len(truth):.1%})"
    )


if __name__ == "__main__":
    main()
