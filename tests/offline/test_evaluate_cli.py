from types import SimpleNamespace

from name_that_move.offline import evaluate_cli


def test_evaluate_cli_quotes_numeric_class_labels(monkeypatch, capsys):
    result = SimpleNamespace(
        session_dir="session4",
        n_windows=3,
        label_counts={"1": 2, "9": 1},
        mean_confidence=0.8,
        accuracy=None,
    )
    monkeypatch.setattr(evaluate_cli, "evaluate_session", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        "sys.argv",
        [
            "name-that-move-evaluate",
            "--session-dir",
            "session4",
            "--model-dir",
            "models",
        ],
    )

    evaluate_cli.main()

    assert "Predicted labels: '1'=2, '9'=1" in capsys.readouterr().out
