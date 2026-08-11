# MiniRocket On the Fly

Reusable MiniRocket-based motion classification for multichannel IMU time
series. The package supports dataset preparation, offline augmentation,
training, model persistence, and batch or single-window inference.

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

Continuous streams use `(timesteps, channels)` and can be segmented into
model-ready windows:

```python
from minirocket_on_the_fly import IMUWindowConfig, make_windows

config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
X = make_windows(continuous_samples, config=config)
print(X.shape)  # (N, 6, 96)
```

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

## Train a model

```python
from minirocket_on_the_fly import (
    extract_features,
    make_dataset,
    save_artifacts,
    train,
)

X, y, splits = make_dataset(
    base_path="path/to/data",
    n_aug=2,
    val_fraction=0.2,
    random_seed=42,
)
X_features, feature_extractor = extract_features(X, splits)
learner = train(X_features, y, splits, epochs=30, lr=None)
save_artifacts(feature_extractor, learner, X, output_dir="models", tag="demo")
```

`make_dataset()` splits the original samples first and augments only the
training set, preventing augmented copies from leaking into validation.

## Run inference

```python
from minirocket_on_the_fly import load_model, load_segment, predict

feature_extractor, learner = load_model("models", tag="demo")
window = load_segment("path/to/window.pkl")
probabilities, labels = predict(window, feature_extractor, learner)
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

Runnable examples are in [`notebooks/`](notebooks/).

## Development checks

```bash
python -m pytest
ruff check .
python -m build
```

## License

[MIT](LICENSE)
