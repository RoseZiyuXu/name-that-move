from types import SimpleNamespace

import numpy as np
import pytest

from name_that_move.offline import train_cli


def test_resolve_validation_sessions_accepts_unique_names_and_paths():
    class_folders = {
        "water": ("water/water_session_01", "water/water_session_02"),
        "desk": ("desk/desk_session_01", "desk/desk_session_02"),
    }

    resolved = train_cli.resolve_validation_sessions(
        class_folders,
        ["water_session_02", "desk/desk_session_02"],
    )

    assert resolved == {
        "water/water_session_02",
        "desk/desk_session_02",
    }


def test_resolve_validation_sessions_reports_ambiguous_name():
    class_folders = {
        "water": ("water/session_01", "water/session_02"),
        "desk": ("desk/session_01", "desk/session_02"),
    }

    with pytest.raises(ValueError, match="ambiguous"):
        train_cli.resolve_validation_sessions(class_folders, ["session_02"])


def test_train_cli_runs_recorded_session_workflow(monkeypatch, capsys, tmp_path):
    class_folders = {
        "water": ("water/water_session_01", "water/water_session_02"),
        "desk": ("desk/desk_session_01", "desk/desk_session_02"),
    }
    X = np.ones((4, 6, 96), dtype=np.float32)
    y = np.array(["water", "desk", "water", "desk"])
    splits = (np.array([0, 1]), np.array([2, 3]))
    workflow = SimpleNamespace(
        np=np,
        torch=SimpleNamespace(manual_seed=lambda seed: None),
        IMUWindowConfig=lambda **kwargs: "config",
        discover_recording_sessions=lambda *args: class_folders,
        make_session_dataset=lambda *args, **kwargs: (X, y, splits),
        extract_features=lambda *args, **kwargs: (
            np.ones((4, 8, 1)),
            "extractor",
        ),
        train=lambda *args, **kwargs: "learner",
        save_artifacts=lambda *args, **kwargs: None,
        load_model=lambda *args, **kwargs: ("extractor", "learner"),
        predict=lambda *args, **kwargs: (None, np.array(["water", "desk"])),
    )
    monkeypatch.setattr(train_cli, "_load_workflow", lambda: workflow)
    monkeypatch.setattr(
        "sys.argv",
        [
            "name-that-move-train",
            "--dataset-dir",
            str(tmp_path),
            "--label",
            "water",
            "--label",
            "desk",
            "--validation-session",
            "water_session_02",
            "--validation-session",
            "desk_session_02",
            "--output-dir",
            str(tmp_path / "model"),
            "--model-tag",
            "water_desk_v0",
        ],
    )

    train_cli.main()

    output = capsys.readouterr().out
    assert "water/water_session_01" in output
    assert "desk/desk_session_02" in output
    assert "Held-out validation: 2/2 (100.0%)" in output
