# motion-clf

Time-series motion classification using [MiniRocket](https://github.com/timeseriesAI/tsai).

## Update & Notice
- As of 2026.05.06 1:00 PM Wednesday, May 6, 2026 (PDT), everything works well on Zoe's machine. It might need further tests. Please start with "pip install git+https://github.com/caizhuodi/DXARTS_TSCLF.git".
- Tsai package changed the path of "get_minirocket_features".

## Install

```bash
pip install motion-clf          # from PyPI once published
# or, in development mode:
pip install -e ".[dev]"
```

## Quickstart

```python
import tsai.models.utils as _utils
from tsai.models.MINIROCKET_Pytorch import get_minirocket_features
_utils.get_minirocket_features = get_minirocket_features
from motion_clf import make_dataset, extract_features, train, save_artifacts

# 1. Load raw segments, apply offline augmentation, and create train/val splits
X, y, splits = make_dataset(
    base_path="path/to/data",
    n_aug=2,
    val_fraction=0.2,
    random_seed=42,
)

# 2. Fit MiniRocketFeatures on the training split and extract features
X_feat, mrf = extract_features(X, splits)

# 3. Train a linear classifier head (pass lr=None to auto-detect via lr_find)
learn = train(X_feat, y, splits, epochs=30, lr=None)

# 4. Save feature extractor + learner + input-shape metadata
save_artifacts(mrf, learn, X, output_dir="./models", tag="my_run")
```

See `notebooks/train_example.ipynb` for a runnable version.

## Data layout

```
base_path/
    all_0_negative/     *.pkl   → class 0
    all_1_beginhand/    *.pkl   → class 1
    ...
```

Each `.pkl` must deserialise to a sequence whose first element is a
`numpy.ndarray` of shape `(n_channels, n_timesteps)` — e.g. `(24, 96)`.

## API

| Function | Description |
|---|---|
| `load_segments(base_path, file_names)` | Load raw PKL segments into `(X, y)` |
| `augment_segments(X, y, n_aug, ...)` | Offline AddNoise + TimeWarp augmentation |
| `make_dataset(base_path, ...)` | Combined load → augment → split helper |
| `extract_features(X, splits, chunksize)` | Fit MiniRocket and extract features |
| `train(X_feat, y, splits, epochs, ...)` | Train linear head with one-cycle LR |
| `save_artifacts(mrf, learn, X, ...)` | Save model weights and metadata |
