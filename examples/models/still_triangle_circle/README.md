# Still-Triangle-Circle example model

This is the deliberately public reference model used by the Name That Move
tutorial and TouchDesigner example.

- Model tag: `still_triangle_circle_v0`
- Classes: `still`, `triangle`, and `circle`
- Input: six channels, 96 timesteps, 48 Hz, 2-second windows
- Training data: Sessions 1-2 of the public example dataset
- Held-out workflow check: Session 3

The published artifacts were reloaded and evaluated against all 90 Session 3
windows: 86 were classified correctly (95.6%). This is a reproducibility check
for this particular same-performer, same-device split, not a general benchmark.

The three files store the MiniRocket feature extractor, the exported learner,
and the versioned input-shape contract. Load all three together through
`name_that_move.load_model`; do not open the learner pickle directly.

This small model is intended for installation checks, tutorials, and the
end-to-end demo. Its held-out result measures a later recording by the same
performer using the same sensor and placement. It is not evidence of accuracy
across performers, devices, placements, or movement styles.

Retrain it from the repository root with:

```bash
python examples/train_example_model.py
```

By default, retraining writes to the Git-ignored
`artifacts/models/example_data/` so it does not silently overwrite this
published reference model.

Model files may use pickle internally. Load them only from this trusted
repository. Unless otherwise noted, these files are provided under the
repository's [MIT License](../../../LICENSE).
