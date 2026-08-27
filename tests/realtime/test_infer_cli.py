from types import SimpleNamespace

from name_that_move.realtime import infer_cli


def test_live_cli_reports_local_loading_and_sections(monkeypatch, capsys):
    predictor = SimpleNamespace(predict=lambda window: window)
    monkeypatch.setattr(
        infer_cli,
        "build_predictor",
        lambda **kwargs: predictor,
    )

    class FakePipeline:
        def __init__(self, **kwargs):
            del kwargs

        def run_forever(self):
            return None

    monkeypatch.setattr(infer_cli, "RealtimePipeline", FakePipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "name-that-move-live",
            "--model-dir",
            "models",
        ],
    )

    infer_cli.main()

    output = capsys.readouterr().out
    assert "\nStarting Name That Move live inference...\n" in output
    assert "\nLoading local model...\nLocal model ready.\n" in output
    assert "\nStarting OSC receiver...\n" in output
    assert "\nLive inference configuration\n" in output
    assert "\nExpected OSC addresses\n" in output
