# Name That Move

**IMU motion classification for dance, choreography, and interactive
performance — powered by MiniRocket.**

Name That Move turns multichannel IMU time series into labeled movements. The
package supports dataset preparation, offline augmentation, training, model
persistence, batch inference, and real-time OSC recording.

The package is sensor- and transmitter-agnostic at its six-channel data
boundary. Movesense Sport and Holon.ist are the tested reference combination,
not requirements; users may substitute another IMU sensor, OSC sender, or
acquisition workflow that satisfies the documented input contract.

> **Project status:** Alpha. The core pipeline works, while the public API and
> cross-platform installation are still being tested.

## Input convention

Model-ready arrays use the shape `(n_samples, n_channels, n_timesteps)`. The
default configuration matches the current performance pipeline: one six-axis
IMU represented on a uniform 48 Hz model-input timeline in two-second windows.
A single window therefore has shape `(6, 96)`, and a batch has shape
`(N, 6, 96)`.

Sensor measurements and BLE/OSC messages may arrive on a non-uniform timeline.
For real-time use, the package samples the latest complete six-channel state
at the configured rate. The 48 Hz default is therefore a configurable package
setting, not a claim about the sensor's native output rate. Training and
inference must use the same configuration.

The default channel order is `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`,
`gyro_z`. Sampling settings remain configurable, but inference data must match
the sample rate, duration, channel order, units, and shape used to train the
loaded model.

Pass the same `IMUWindowConfig` through loading, training, model saving, and
inference. Configuration mismatches fail before MiniRocket feature extraction
and report the expected and received values.

Continuous streams use `(timesteps, channels)` and can be segmented into
model-ready windows:

```python
from name_that_move import IMUWindowConfig, make_windows

config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
X = make_windows(continuous_samples, config=config)
print(X.shape)  # (N, 6, 96)
```

## Workflow architecture

Shared preprocessing and inference components support two workflows:

- **Offline:** recorded files → validation/windowing → dataset preparation →
  feature extraction → training/evaluation → saved model.
- **Real-time:** OSC stream → latest-value buffer → fixed-rate window →
  asynchronous recording and optional non-blocking inference → media output.

Workflow-specific code lives under `offline/` and `realtime/`. Shared input
configuration, preprocessing, and model inference remain at package level so
the two workflows use the same data contract.

## Installation

Install the current development version from GitHub:

```bash
python -m pip install "git+https://github.com/RoseZiyuXu/name-that-move.git"
```

For local development:

```bash
git clone https://github.com/RoseZiyuXu/name-that-move.git
cd name-that-move
python -m pip install --editable ".[dev]"
```

For recording without development tools:

```bash
python -m pip install ".[realtime]"
```

## Record OSC IMU windows

The recorder preserves the performance prototype's sampling strategy: each
incoming OSC message updates one named channel, and the computer samples the
latest six values at 48 Hz. It waits for all six channels before starting and
saves each completed window on a background thread, so disk writing does not
pause sampling.

```bash
name-that-move-record \
  --label line \
  --session session_01 \
  --imu-id 1 \
  --port 10000
```

The 48 Hz and two-second defaults are configurable. For example, the command
below records two-second windows at 52 Hz, producing `(6, 104)` arrays:

```bash
name-that-move-record --label line --sample-rate 52 --window-duration 2
```

Configure the OSC sender—for example, a phone data-transmitter app—to send to
the computer's local-network IP on port `10000`, using these addresses for the
default IMU ID:

```text
/m/1/acc/x
/m/1/acc/y
/m/1/acc/z
/m/1/gyro/x
/m/1/gyro/y
/m/1/gyro/z
```

For a different sender namespace, add an OSC prefix while preserving the six
channel suffixes. For example, ``--osc-prefix /wearable/right-wrist`` expects
``/wearable/right-wrist/acc/x`` through
``/wearable/right-wrist/gyro/z``.

Generated files are stored under
`artifacts/recordings/<label>/<session>/`. Each `.pkl` contains a `(6, 96)`
window; the matching `.json` records the configuration and lightweight
connection diagnostics. The `artifacts/` directory is ignored by Git.

Use Control-C to stop and flush pending files. The equivalent module command
is:

```bash
python -m name_that_move.realtime.cli --label line
```

Remote inference is a separate optional feature:

```bash
python -m pip install ".[remote]"
```

`InferenceWorker` runs either local or remote prediction outside the sampling
thread. A bounded queue prevents slow inference from creating an ever-growing
backlog; when inference cannot keep up, the newest completed window is
reported as unclassified rather than pausing acquisition.

## Train a model

Record at least two motion classes in separate training and validation
sessions, then train directly from the recorder's
``artifacts/recordings/<label>/<session>/`` layout:

**For every selected motion class, all sessions not supplied through
`--validation-session` are automatically used for training.** Repeat
`--validation-session` to hold out more than one complete session.

```bash
name-that-move-train \
  --dataset-dir artifacts/recordings \
  --label my_move \
  --label your_move \
  --validation-session my_move_session_02 \
  --validation-session your_move_session_02 \
  --output-dir artifacts/models/our_moves \
  --model-tag our_moves_v0
```

This trains on the other discovered sessions, holds out both Session 2 folders,
saves the model, reloads it, and reports held-out validation accuracy. See the
training tutorial for the complete custom-data workflow.

The Python API supports lower-level or programmatic workflows:

```python
from name_that_move import (
    IMUWindowConfig,
    extract_features,
    make_dataset,
    save_artifacts,
    train,
)

config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
X, y, splits = make_dataset(
    base_path="path/to/data",
    n_aug=2,
    val_fraction=0.2,
    random_seed=42,
    config=config,
)
X_features, feature_extractor = extract_features(X, splits, config=config)
learner = train(X_features, y, splits, epochs=30, lr=None)
save_artifacts(
    feature_extractor,
    learner,
    X,
    output_dir="models",
    tag="demo",
    config=config,
)
```

`make_dataset()` splits the original samples first and augments only the
training set, preventing augmented copies from leaking into validation.

### Run the three-class example workflow

For datasets recorded as multiple sessions per class, use
`make_session_dataset()` to hold out complete sessions instead of randomly
splitting related windows. The public example uses Sessions 1–2 for
training and Session 3 for validation:

- Recording round 1 produced `still1`, `triangle1`, and `circle1`.
- Recording round 2 produced the corresponding folders ending in `2`.
- Recording round 3 produced the corresponding folders ending in `3`.

The numeric suffix identifies the recording session, not a separate class.
Holding out all suffix-`3` folders provides cross-session validation and avoids
placing neighboring windows from one recording in both splits. This checks the
package workflow and transfer to a later recording; it is not yet a
cross-performer or cross-device benchmark.

```bash
python examples/train_example_model.py
```

The script trains `still`, `triangle`, and `circle`, saves three model
artifacts under `artifacts/models/example_data/`, reloads them, and runs
inference on the held-out session. The deliberately public source data and
reference model live under `examples/data/still_triangle_circle/` and
`examples/models/still_triangle_circle/`. Retrained outputs remain Git-ignored
so they do not overwrite the published reference accidentally.

## Run inference

```python
from name_that_move import IMUWindowConfig, load_model, load_segment, predict

config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
feature_extractor, learner = load_model(
    "models", tag="demo", expected_config=config
)
window = load_segment("path/to/window.pkl")
probabilities, labels = predict(
    window, feature_extractor, learner, config=config
)
```

## Data layout

```text
data/
├── all_0_negative/
│   └── *.pkl
├── all_1_beginhand/
│   └── *.pkl
└── ...
```

Each `.pkl` file must contain either an array or a sequence whose first item is
an array with shape `(n_channels, n_timesteps)`. Only load pickle files from
sources you trust.

## Public API

| Function | Purpose |
| --- | --- |
| `load_segments` | Load labeled segment files |
| `augment_segments` | Create offline augmented copies |
| `make_dataset` | Load, split, and augment training data |
| `make_session_dataset` | Group class sessions and hold out complete sessions |
| `IMUWindowConfig` | Define sampling rate, duration, and channel order |
| `make_windows` | Segment a continuous IMU stream into model-ready windows |
| `validate_windows` | Validate shape and values and create a float32 batch |
| `extract_features` | Fit MiniRocket and extract features |
| `train` | Train the linear classification head |
| `save_artifacts` | Save the feature extractor and learner |
| `load_model` | Restore saved model artifacts |
| `predict` | Predict labels for one or more windows |

Advanced workflow APIs are available from `name_that_move.offline` and
`name_that_move.realtime`.

Runnable examples are in [`notebooks/`](notebooks/).

## Development checks

```bash
python -m pytest
ruff check .
python -m build
```

## Citation and contributors

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub can
use this file to generate APA and BibTeX citations for the software.

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for development, advising,
conceptualization, and institutional contributions. The project gratefully
acknowledges support from the University of Washington eScience Institute and
the University of Washington Department of Digital Arts and Experimental
Media (DXARTS).

Third-party software and license information are documented in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Contributing and future work

Bug reports, documentation improvements, tested examples, and focused code
contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for issue,
development, testing, data-privacy, and pull-request guidance.

Planned and exploratory work—including direct BLE acquisition, a public
example dataset, release automation, and performance-system integrations—is
tracked in [`ROADMAP.md`](ROADMAP.md). Roadmap items are development ideas, not
commitments to a particular release.

## License

[MIT](LICENSE)
