# MiniRocket On the Fly

Reusable MiniRocket-based motion classification for multichannel IMU time
series. The package supports dataset preparation, offline augmentation,
training, model persistence, and batch or single-window inference.

> **Project status:** Alpha. The core pipeline works, while the public API and
> cross-platform installation are still being tested.

## Input convention

Arrays use the shape `(n_samples, n_channels, n_timesteps)`. A single IMU
window may therefore have shape `(6, 48)`, and a batch of windows `(N, 6, 48)`.

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
