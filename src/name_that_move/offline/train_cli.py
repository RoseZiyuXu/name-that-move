"""Command-line entry point for training from recorded motion sessions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace


def _load_workflow() -> SimpleNamespace:
    """Import the numerical training stack only after arguments are parsed."""
    import numpy as np
    import torch

    from name_that_move.infer import load_model, predict
    from name_that_move.offline.data import (
        discover_recording_sessions,
        make_session_dataset,
    )
    from name_that_move.offline.training import (
        extract_features,
        save_artifacts,
        train,
    )
    from name_that_move.preprocessing import IMUWindowConfig

    return SimpleNamespace(
        np=np,
        torch=torch,
        load_model=load_model,
        predict=predict,
        discover_recording_sessions=discover_recording_sessions,
        make_session_dataset=make_session_dataset,
        extract_features=extract_features,
        save_artifacts=save_artifacts,
        train=train,
        IMUWindowConfig=IMUWindowConfig,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for session-aware training from recorder output."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a saved motion classifier from class/session folders created "
            "by name-that-move-record."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/recordings"),
        help="Recording root containing <label>/<session> folders",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Motion class to include; repeat for multiple classes (default: all)",
    )
    parser.add_argument(
        "--validation-session",
        action="append",
        required=True,
        help=(
            "Session reserved for validation; repeat once per class. Accepts a "
            "session name or a path relative to --dataset-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models/custom_model"),
        help="Directory for the three saved model artifacts",
    )
    parser.add_argument("--model-tag", default="custom_model")
    parser.add_argument("--sample-rate", type=float, default=48.0)
    parser.add_argument("--window-duration", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help="Create N augmented copies of each training window (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_validation_sessions(
    class_folders: Mapping[str, Sequence[str]],
    requested_sessions: Sequence[str],
) -> set[str]:
    """Resolve unique session names or relative paths to discovered paths."""
    available = {
        folder for folders in class_folders.values() for folder in folders
    }
    resolved: set[str] = set()
    for requested in requested_sessions:
        normalized = Path(requested).as_posix().strip("/")
        if normalized in available:
            match = normalized
        else:
            matches = sorted(
                folder for folder in available if Path(folder).name == normalized
            )
            if not matches:
                choices = ", ".join(sorted(available))
                raise ValueError(
                    f"Validation session not found: {requested}. "
                    f"Available sessions: {choices}"
                )
            if len(matches) > 1:
                choices = ", ".join(matches)
                raise ValueError(
                    f"Validation session name is ambiguous: {requested}. "
                    f"Use one of these relative paths: {choices}"
                )
            match = matches[0]
        if match in resolved:
            raise ValueError(f"Validation session was repeated: {requested}")
        resolved.add(match)
    return resolved


def main() -> None:
    """Train, save, reload, and validate a model from recorded sessions."""
    args = build_parser().parse_args()
    print("\nStarting Name That Move model training...", flush=True)
    workflow = _load_workflow()
    try:
        class_folders = workflow.discover_recording_sessions(
            args.dataset_dir, args.labels
        )
        if len(class_folders) < 2:
            raise ValueError("Training requires at least two motion classes")
        validation_folders = resolve_validation_sessions(
            class_folders, args.validation_session
        )
        config = workflow.IMUWindowConfig(
            sample_rate_hz=args.sample_rate,
            window_duration_s=args.window_duration,
        )
        workflow.np.random.seed(args.seed)
        workflow.torch.manual_seed(args.seed)

        print("\nSession split")
        for label, sessions in class_folders.items():
            training = [s for s in sessions if s not in validation_folders]
            validation = [s for s in sessions if s in validation_folders]
            print(f"  {label}")
            print(f"    Training: {', '.join(training) or 'none'}")
            print(f"    Validation: {', '.join(validation) or 'none'}")

        X, y, splits = workflow.make_session_dataset(
            args.dataset_dir,
            class_folders,
            validation_folders=validation_folders,
            n_aug=args.augment,
            random_seed=args.seed,
            config=config,
        )
        X_features, feature_extractor = workflow.extract_features(
            X, splits, chunksize=args.chunksize, config=config
        )
        learner = workflow.train(
            X_features,
            y,
            splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            show_graph=False,
        )
        workflow.save_artifacts(
            feature_extractor,
            learner,
            X,
            output_dir=args.output_dir,
            tag=args.model_tag,
            config=config,
        )

        loaded_extractor, loaded_learner = workflow.load_model(
            args.output_dir,
            tag=args.model_tag,
            expected_config=config,
        )
        validation_indices = splits[1]
        _, predictions = workflow.predict(
            X[validation_indices],
            loaded_extractor,
            loaded_learner,
            config=config,
        )
        truth = y[validation_indices]
        predictions = workflow.np.asarray(predictions, dtype=str)
        correct = int(workflow.np.sum(predictions == truth))
        print("\nTraining complete")
        print(f"  Model directory: {args.output_dir}")
        print(f"  Model tag: {args.model_tag}")
        print(
            f"  Held-out validation: {correct}/{len(truth)} "
            f"({correct / len(truth):.1%})"
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None


if __name__ == "__main__":
    main()
