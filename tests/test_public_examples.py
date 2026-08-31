from pathlib import Path

from name_that_move import load_segment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY_ROOT / "examples" / "data" / "still_triangle_circle"
MODEL = REPOSITORY_ROOT / "examples" / "models" / "still_triangle_circle"
SESSIONS = {
    "still1",
    "still2",
    "still3",
    "triangle1",
    "triangle2",
    "triangle3",
    "circle1",
    "circle2",
    "circle3",
}
TAG = "still_triangle_circle_v0"


def test_public_example_dataset_is_complete_and_loadable():
    assert {path.name for path in DATASET.iterdir() if path.is_dir()} == SESSIONS
    for session in SESSIONS:
        windows = sorted((DATASET / session).glob("*.pkl"))
        assert len(windows) == 30
        assert load_segment(windows[0]).shape == (6, 96)


def test_public_example_model_contains_complete_artifact_set():
    expected = {
        f"MRF-{TAG}.pt",
        f"MRL-{TAG}.pkl",
        f"input_shape-{TAG}.pt",
    }
    assert expected.issubset({path.name for path in MODEL.iterdir()})
