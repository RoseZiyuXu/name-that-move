# Name That Move

**Wearable-sensor movement recognition for dance, choreography, and
interactive performance!**

Name That Move is an open-source toolkit for recording labeled six-axis IMU
windows, training personalized movement classifiers, and using predictions in
offline analysis or live creative systems. The example training pipeline uses
[MiniRocket](https://arxiv.org/abs/2012.08791), a fast time-series feature
transform, with a linear classifier.

The package uses a consistent six-channel data contract across recording,
training, and inference. Movesense Sport and Holon.ist provide the tested OSC
reference workflow, but other sensors and transmitters can be used when they
produce the same model-ready data.

> **Questions and feedback are welcome.** Questions, corrections, unsuccessful
> experiments, and suggestions are all valuable contributions. If something is
> unclear, please [open a GitHub Issue](https://github.com/RoseZiyuXu/name-that-move/issues).
> If GitHub is unfamiliar or your message involves private participant data or
> unpublished artistic material, please do not hesitate to
> [contact Rose](https://github.com/RoseZiyuXu).

## Quick start

Name That Move supports Python 3.9 or newer; Python 3.11 matches the continuous
integration environment.

```bash
conda create -n name-that-move python=3.11
conda activate name-that-move
python -m pip install "git+https://github.com/RoseZiyuXu/name-that-move.git"
```

See the complete [installation guide](https://name-that-move.readthedocs.io/en/latest/installation.html)
for local editable installation, optional dependencies, and verification.

## Documentation

The [full documentation](https://name-that-move.readthedocs.io/en/latest/)
introduces the tools and follows the package from sensor setup through creative
application.

| Goal | Guide |
| --- | --- |
| Learn what Python, the terminal, and commands do | [Before you begin](https://name-that-move.readthedocs.io/en/latest/before_you_begin.html) |
| Understand the supported sensor and OSC setup | [Hardware and data flow](https://name-that-move.readthedocs.io/en/latest/hardware_setup.html) |
| Check channel order, shape, units, and sampling | [IMU data contract](https://name-that-move.readthedocs.io/en/latest/data_contract.html) |
| Review command options and shared settings | [Configuration quick reference](https://name-that-move.readthedocs.io/en/latest/configuration.html) |
| Record labeled IMU windows | [Record labeled IMU windows](https://name-that-move.readthedocs.io/en/latest/recording.html) |
| Train the bundled example or your own model | [Train a model](https://name-that-move.readthedocs.io/en/latest/training.html) |
| Run offline or live, local or remote inference | [Run inference](https://name-that-move.readthedocs.io/en/latest/inference.html) |
| Connect predictions to TouchDesigner and other media systems | [Build a real-time performance application](https://name-that-move.readthedocs.io/en/latest/realtime.html) |

The repository includes a deliberately public three-class tutorial dataset,
saved reference model, training script, and TouchDesigner patch under
[`examples/`](examples/).

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
`name_that_move.realtime`. See the complete
[API reference](https://name-that-move.readthedocs.io/en/latest/api.html).

## Citation and contributors

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub can
use this file to generate APA and BibTeX citations for the software. See the
[citation guide](https://name-that-move.readthedocs.io/en/latest/citing.html)
for the project's related research publications.

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
development, testing, data-privacy, and pull-request guidance. Development
ideas are tracked in [`ROADMAP.md`](ROADMAP.md).

## License

[MIT](LICENSE)
