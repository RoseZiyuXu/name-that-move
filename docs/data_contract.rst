IMU data contract
=================

Model-ready arrays use channel-first shape:

.. code-block:: text

   (n_samples, n_channels, n_timesteps)

With the default configuration:

* ``n_channels = 6``;
* ``sample_rate_hz = 48``;
* ``window_duration_s = 2``; and
* ``n_timesteps = 48 × 2 = 96``.

A single model-ready window has shape ``(6, 96)``. A batch has shape
``(N, 6, 96)``. Some recorded pickle files contain a one-window batch with
shape ``(1, 6, 96)``; package loaders normalize both representations.

Default channel order
---------------------

.. code-block:: text

   acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

Use one :class:`name_that_move.preprocessing.IMUWindowConfig` consistently for
recording, loading, feature extraction, model saving, and inference. The
package fails early when shape or saved-model metadata do not match.

Session-aware evaluation
------------------------

A recording session is one separate acquisition period or recording round.
Randomly splitting neighboring windows can place highly similar samples from
the same recording in both training and validation. This can create
session-specific similarity leakage and an overly optimistic result.

The example workflow trains on Sessions 1 and 2 and holds out all of Session 3.
This cross-session validation gives stronger evidence that the workflow
transfers to a later recording. It is not a cross-performer, cross-device, or
general accuracy benchmark.

Security note
-------------

Pickle files can execute arbitrary code while loading. Only load ``.pkl``
files from sources you trust.
