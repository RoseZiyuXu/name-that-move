# Name That Move examples

The repository includes a deliberately public three-class tutorial dataset and
reference model:

- [`data/still_triangle_circle/`](data/still_triangle_circle/): 270 recorded
  IMU windows for `still`, `triangle`, and `circle`
- [`models/still_triangle_circle/`](models/still_triangle_circle/): the matching
  ready-to-use model with tag `still_triangle_circle_v0`
- [`touchdesigner/`](touchdesigner/): a downstream creative-coding patch that
  receives predictions, filters confidence, and visualizes accepted labels

From the repository root, retrain and validate a fresh model with:

```bash
python examples/train_example_model.py
```

Or run the published model directly with live OSC input and TouchDesigner
output:

```bash
name-that-move-live \
  --model-dir examples/models/still_triangle_circle \
  --model-tag still_triangle_circle_v0 \
  --ip 0.0.0.0 \
  --port 10000 \
  --imu-id 1 \
  --sample-rate 48 \
  --window-duration 2 \
  --startup-timeout 2 \
  --touchdesigner-ip 127.0.0.1 \
  --touchdesigner-port 8000 \
  --touchdesigner-path /sensor/1
```

These tutorial assets are distributed through the GitHub repository rather
than installed into Python's `site-packages`. Clone or download the repository
before running these path-based commands.
