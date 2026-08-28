# Name That Move Development Roadmap

Name That Move is alpha research software. This roadmap records development
ideas and priorities; it is not a promise that every item will be implemented
or included in a particular release. Priorities may change through real-world
use, research needs, maintainer capacity, and community contributions.

## Current foundation

The package currently provides:

- one configurable six-axis IMU window contract shared by training and
  inference;
- session-aware dataset preparation and a small real-data example workflow;
- saved-model loading and offline local or remote evaluation;
- live local or remote inference from OSC input;
- non-blocking recording and inference workers;
- optional TouchDesigner prediction output;
- tests on Linux, macOS, and Windows; and
- Sphinx documentation and Read the Docs pull-request builds.

## Near-term end-to-end work

### Dual OSC and BLE acquisition

Preserve the tested Movesense-to-phone-to-OSC workflow and add direct
Movesense-to-laptop BLE as an optional transport. The BLE work includes:

- testing an official Movesense GATT Sensor Data firmware on one recoverable
  sensor;
- adapting the official Python reader without duplicating downstream logic;
- verifying packet decoding, timestamps, axis order, units, and available
  native sample rates;
- converting both OSC and BLE input into the same named six-channel interface;
- adding connection, malformed-packet, timeout, and disconnect tests; and
- documenting range, firmware, compatibility, and recovery tradeoffs.

BLE should replace only the acquisition adapter. Window construction,
recording, local or remote inference, and TouchDesigner output should remain
shared.

### Release and installation

- Test the built distributions through TestPyPI in a clean environment.
- Configure PyPI Trusted Publishing with a tag-triggered workflow and a manual
  approval environment.
- Publish an initial alpha release and document version and release practices.
- Continue testing installation and commands on the three supported operating
  systems.

### End-to-end tutorial and media example

- Complete one reproducible walkthrough from sensor setup through recording,
  training, saved-model evaluation, live inference, and media output.
- Add a small TouchDesigner example patch that receives label and confidence
  and maps predictions to a simple audiovisual response.
- Keep command-line feedback and fail-fast connection guidance suitable for
  workshop and performance use.

## Dataset and evaluation development

- Release a deliberately public, consent-cleared example dataset with simple
  classes such as stillness, triangle, and circle.
- Release a compatible small example model and known-input, known-prediction
  regression test.
- Expand evaluation across recording sessions, performers, sensor placements,
  and devices without overstating small smoke-test results.
- Record sensor timestamps and connection diagnostics when available, and
  evaluate timestamp-aware resampling for irregular streams.
- Document sensor orientation, acceleration and gyroscope units, class
  protocols, and model limitations.

## Longer-term development ideas

- adapters for additional IMU sensors and data transports;
- multiple simultaneous sensors and configurable sensor placement;
- additional time-series feature extractors and model backends;
- richer confidence calibration, evaluation reports, and model metadata;
- reusable integrations for TouchDesigner and other performance systems; and
- examples contributed by dance, music, interactive-media, and wearable-
  computing users.

## Help shape the roadmap

Use [GitHub Issues](https://github.com/RoseZiyuXu/name-that-move/issues) to
describe a problem, propose a development idea, or report an unsuccessful
experiment. Explain the intended user and workflow, required hardware or data,
and how the idea could be tested. See the
[contribution guide](https://github.com/RoseZiyuXu/name-that-move/blob/main/CONTRIBUTING.md)
before preparing a pull request.
