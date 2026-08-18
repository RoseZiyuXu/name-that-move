# MiniRocket On the Fly

Reusable MiniRocket-based motion classification for multichannel IMU time
series. The package supports dataset preparation, offline augmentation,
training, model persistence, batch inference, and real-time OSC recording.

> **Project status:** Alpha. The core pipeline works, while the public API and
> cross-platform installation are still being tested.

## Input convention

Model-ready arrays use the shape `(n_samples, n_channels, n_timesteps)`. The
default configuration matches the current performance pipeline: one six-axis
IMU sampled at 48 Hz in two-second windows. A single window therefore has
shape `(6, 96)`, and a batch has shape `(N, 6, 96)`.

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
from minirocket_on_the_fly import IMUWindowConfig, make_windows

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
python -m pip install "git+https://github.com/RoseZiyuXu/Minirocket_OnTheFly.git"
```

For local development:

```bash
git clone https://github.com/RoseZiyuXu/Minirocket_OnTheFly.git
cd Minirocket_OnTheFly
python -m pip install ".[dev]"
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
minirocket-record \
  --label line \
  --session session_01 \
  --imu-id 2 \
  --port 10000
```

The 48 Hz and two-second defaults are configurable. For example, the command
below records two-second windows at 52 Hz, producing `(6, 104)` arrays:

```bash
minirocket-record --label line --sample-rate 52 --window-duration 2
```

Configure the phone to send to the computer's local-network IP on port
`10000`, using these addresses for the default IMU ID:

```text
/m/2/acc/x
/m/2/acc/y
/m/2/acc/z
/m/2/gyro/x
/m/2/gyro/y
/m/2/gyro/z
```

Generated files are stored under
`artifacts/recordings/<label>/<session>/`. Each `.pkl` contains a `(6, 96)`
window; the matching `.json` records the configuration and lightweight
connection diagnostics. The `artifacts/` directory is ignored by Git.

Use Control-C to stop and flush pending files. The equivalent module command
is:

```bash
python -m minirocket_on_the_fly.realtime.cli --label line
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

```python
from minirocket_on_the_fly import (
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

## Run inference

```python
from minirocket_on_the_fly import IMUWindowConfig, load_model, load_segment, predict

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
| `IMUWindowConfig` | Define sampling rate, duration, and channel order |
| `make_windows` | Segment a continuous IMU stream into model-ready windows |
| `validate_windows` | Validate shape and values and create a float32 batch |
| `extract_features` | Fit MiniRocket and extract features |
| `train` | Train the linear classification head |
| `save_artifacts` | Save the feature extractor and learner |
| `load_model` | Restore saved model artifacts |
| `predict` | Predict labels for one or more windows |

Advanced workflow APIs are available from `minirocket_on_the_fly.offline` and
`minirocket_on_the_fly.realtime`.

Runnable examples are in [`notebooks/`](notebooks/).

## Development checks

```bash
python -m pytest
ruff check .
python -m build
```

## License

[MIT](LICENSE)
