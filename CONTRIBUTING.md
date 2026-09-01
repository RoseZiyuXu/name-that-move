# Contributing to Name That Move

Thank you for helping make movement-recognition tools more useful to artists,
researchers, students, performers, and software developers. Contributions of
different sizes and backgrounds are welcome.

## Ways to contribute

You can:

- report a reproducible bug;
- suggest a workflow, sensor, model, or documentation improvement;
- clarify documentation or add a tested example;
- add or improve tests;
- propose a focused code change; or
- share how Name That Move behaves in a new performance or research setting.

Use [GitHub Issues](https://github.com/RoseZiyuXu/name-that-move/issues) for
bugs and feature requests so that answers can help future users. If a report
cannot be public because it involves private participant data, unpublished
artistic material, or a security concern, contact the maintainer privately
using the contact information on the maintainer's GitHub profile.

## Report a useful bug

Search existing issues first. A new report should include, when relevant:

- operating system and Python version;
- Name That Move version or Git commit;
- installation method and selected optional dependencies;
- whether the workflow is offline or real-time, local or remote, and OSC or
  another acquisition transport;
- the smallest command or code example that reproduces the problem;
- the complete short error message or traceback;
- input shape, channel order, sampling rate, window duration, and units; and
- whether the problem is consistent or intermittent.

Do not upload private recordings, credentials, model-server URLs, participant
information, or performance data without permission. A small synthetic example
that has the same shape and failure is preferred.

## Propose a feature

Describe the user and workflow problem before proposing an implementation.
Explain what success would look like, what data contract it requires, and
whether it affects training, saved-model metadata, offline inference, or
real-time inference. Larger development ideas should be discussed in an issue
before a pull request is opened. The current direction is summarized in the
[development roadmap](https://github.com/RoseZiyuXu/name-that-move/blob/main/ROADMAP.md).

## Development setup

Python 3.11 matches the continuous-integration environment.

```bash
git clone https://github.com/RoseZiyuXu/name-that-move.git
cd name-that-move
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev,docs,realtime,remote]"
```

Run the same core checks used by continuous integration:

```bash
python -m pytest
ruff check .
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

## Pull requests

Keep a pull request focused enough to review and test. Before submitting it:

1. Add or update tests for behavior changes.
2. Update docstrings and user documentation when an interface changes.
3. Preserve the shared IMU data contract across offline and real-time paths.
4. Do not commit local datasets, trained models, credentials, build outputs,
   or participant-identifiable material.
5. Run the checks above and explain any check that cannot be run locally.
6. Describe the user-visible result, scientific assumptions, and known limits.

Code contributions are accepted under the repository's MIT License. If a
change copies or adapts third-party material, identify its source and license
and update `THIRD_PARTY_LICENSES.md` when required.

## Reproducibility and research claims

Distinguish a software smoke test from a scientific benchmark. Report how
recording sessions, performers, devices, and classes were split. Avoid random
window splits when neighboring windows from one recording could appear in both
training and validation. Do not describe one held-out session as evidence of
general performance across people or devices.

## AI-assisted contributions

AI tools may assist development, but the human contributor remains responsible
for every submitted change. Review generated code and prose, run the relevant
tests, verify technical claims, and check licensing and attribution. Do not
send private recordings, credentials, unpublished participant information, or
restricted source code to an external AI service. Disclose substantial
AI-assisted code or documentation in the pull-request description when it
would help reviewers understand provenance or verification.

## A welcoming project

Questions and first contributions are welcome. Clear bug reports, documentation
corrections, unsuccessful experiment reports, and small improvements are all
valuable. Please communicate respectfully and make assumptions and limitations
visible so that collaborators from different technical and artistic fields can
participate.
