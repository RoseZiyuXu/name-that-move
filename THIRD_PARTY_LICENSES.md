# Third-Party Software Notices

Name That Move is licensed under the MIT License. It depends on
third-party software that remains subject to its own license terms. Installing
this package may install those dependencies separately; their licenses are not
replaced by the MIT License used for this repository.

## tsai

- Project: [timeseriesAI/tsai](https://github.com/timeseriesAI/tsai)
- License: [Apache License 2.0](https://github.com/timeseriesAI/tsai/blob/main/LICENSE)
- Use in this project: MiniRocket feature extraction, model construction,
  training utilities, and model loading

Name That Move imports and uses `tsai` as a dependency. `tsai` and its
contributors retain their original copyright and license notices. Nothing in
the MIT License for Name That Move changes the Apache License 2.0 terms
that apply to `tsai`.

The Apache License 2.0 permits redistribution under different terms when its
conditions are followed. Relevant conditions include providing the Apache
License, retaining applicable attribution notices, identifying modified
Apache-licensed files, and preserving applicable content from an upstream
`NOTICE` file.

## Release audit

Before a public release, the maintainers should:

1. Confirm whether any upstream source code was copied or adapted rather than
   only imported as a dependency.
2. Preserve all notices required by the applicable upstream licenses.
3. Review the licenses of every direct runtime dependency listed in
   `pyproject.toml`.
4. Update this document whenever dependencies or their license terms change.

This notice is provided for attribution and project documentation and is not
legal advice.
