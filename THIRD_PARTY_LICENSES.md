# Third-Party Notices

Name That Move is licensed under the MIT License. It depends on
third-party software that remains subject to its own license terms. Installing
this package may install those dependencies separately; their licenses are not
replaced by the MIT License used for this repository.

## Published research figures

The documentation reproduces two published research figures under their own
license terms:

- `docs/_static/human_machine_ritual_figure3.jpg` reproduces Figure 3 from
  Zhuodi Cai, Ziyu Xu, and Juan Pampin, *Human-Machine Ritual: Synergic
  Performance through Real-Time Motion Recognition* (2025),
  [arXiv:2511.02351](https://arxiv.org/abs/2511.02351).
- `docs/_static/creativity_generativity_figure2.jpeg` reproduces Figure 2 from
  Ziyu Xu and Zhuodi Cai, *Creativity ≠ Generativity: A Case Study of Attentive
  Machine Learning in Dance Performance* (2026),
  [DOI:10.1145/3816094](https://doi.org/10.1145/3816094).

Both figures are distributed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), not
under the repository's MIT License. Reuse must follow the attribution,
noncommercial-use, and no-derivatives requirements of that license.

## Audited runtime dependencies

The following table covers the direct runtime and optional dependencies in
`pyproject.toml`, plus `fastai`, an important dependency used by `tsai` and
the saved-model workflow. It was last reviewed on 2026-08-26.

| Package | License | Use in Name That Move |
| --- | --- | --- |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause, with separately licensed bundled components | IMU arrays and numerical operations |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-3-Clause main project, with separately licensed bundled components | MiniRocket tensors, model weights, and local inference |
| [tsai](https://github.com/timeseriesAI/tsai/blob/main/LICENSE) | Apache-2.0 | MiniRocket feature extraction, training, and model loading |
| [fastai](https://github.com/fastai/fastai/blob/main/LICENSE) | Apache-2.0 | Learner training, export, and loading through `tsai` |
| [fastcore](https://github.com/AnswerDotAI/fastcore/blob/main/LICENSE) | Apache-2.0 | Supporting dependency for the fastai/tsai stack |
| [fastprogress](https://github.com/AnswerDotAI/fastprogress/blob/master/LICENSE) | Apache-2.0 | Progress reporting in the fastai/tsai stack |
| [tsaug](https://github.com/arundo/tsaug/blob/master/LICENSE) | Apache-2.0 | Optional offline time-series augmentation |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | BSD-3-Clause | Supporting machine-learning utilities |
| [sktime](https://github.com/sktime/sktime/blob/main/LICENSE) | BSD-3-Clause | Supporting time-series dependency |
| [tqdm](https://github.com/tqdm/tqdm/blob/master/LICENCE) | MPL-2.0 and MIT | Offline data-loading progress bars |
| [python-osc](https://github.com/attwad/python-osc/blob/main/LICENSE.txt) | Unlicense | Optional real-time OSC input and output |
| [Requests](https://github.com/psf/requests/blob/main/LICENSE) | Apache-2.0 | Optional remote HTTP inference |

These packages are installed separately by the user's package manager; their
source code is not included in the Name That Move wheel. Their licenses apply
to their own files and do not replace the MIT License for Name That Move.

The MPL-2.0 terms covering parts of `tqdm` are file-level weak copyleft. Using
an unmodified, separately installed `tqdm` package does not require unrelated
Name That Move source files to adopt the MPL. If future work copies or modifies
MPL-covered `tqdm` files, those files must remain available under MPL-2.0 and
retain the required notices.

Apache-2.0 and BSD-licensed packages likewise retain their original copyright
and license notices. If a future release bundles dependency code, a wheel,
standalone executable, container image, or modified upstream files, that
distribution must also carry the applicable license and NOTICE materials for
the components it redistributes.

## Release audit

Before a public release, the maintainers should:

1. Confirm whether any upstream source code was copied or adapted rather than
   only imported as a dependency.
2. Preserve all notices required by the applicable upstream licenses.
3. Recheck the licenses of every direct runtime dependency, especially after
   changing version constraints.
4. Update this document whenever dependencies or their license terms change.

This notice is provided for attribution and project documentation and is not
legal advice.
