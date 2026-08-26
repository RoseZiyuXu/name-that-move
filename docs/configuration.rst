Configuration quick reference
=============================

Use this page when adapting Name That Move to a new sensor, movement dataset,
or performance system. A parameter being configurable does **not** always mean
it can change after training: model-input settings must remain aligned across
recording, training, saved-model loading, and inference.

Must match from recording through inference
-------------------------------------------

These settings define the model's input. Choose them before collecting a
dataset and reuse the same :class:`~name_that_move.IMUWindowConfig` throughout
the workflow.

.. list-table:: Model-input configuration
   :widths: 22 18 26 34
   :header-rows: 1

   * - Parameter
     - Default
     - Where to set it
     - Alignment rule
   * - ``sample_rate_hz``
     - ``48``
     - ``IMUWindowConfig`` or ``--sample-rate``
     - Must match recording, training, saved metadata, and inference.
   * - ``window_duration_s``
     - ``2``
     - ``IMUWindowConfig`` or ``--window-duration``
     - Must match throughout; together with sample rate it determines the
       timestep count.
   * - ``channel_names`` and order
     - ``acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z``
     - ``IMUWindowConfig`` in the Python API
     - Names, number, and order must match throughout.
   * - Units and scaling
     - Sensor-dependent
     - Sensor/app configuration and preprocessing
     - Not automatically detected. Use the same units and scaling for training
       and inference.
   * - Sensor placement and orientation
     - Right wrist, upright for the example model
     - Physical recording protocol
     - Not automatically detected. Reproduce the training placement during
       inference.

The default configuration produces one ``(6, 96)`` window. Changing the sample
rate or duration is supported when their product is a whole number, but an
already trained model cannot accept the new shape unless it was trained and
saved with that configuration.

.. code-block:: python

   from name_that_move import IMUWindowConfig

   config = IMUWindowConfig(
       sample_rate_hz=48,
       window_duration_s=2,
   )

Configure for each recording setup
----------------------------------

These settings control acquisition and file organization. They may change
without retraining, provided that the resulting windows still satisfy the
model-input contract above.

.. list-table:: Recording configuration
   :widths: 22 18 28 32
   :header-rows: 1

   * - Parameter
     - Default
     - Where to set it
     - What must agree
   * - ``imu_id``
     - ``1``
     - ``--imu-id`` or real-time Python API
     - Must match the ID embedded in the phone's OSC address paths.
   * - OSC input ``port``
     - ``10000``
     - ``--port`` or real-time Python API
     - Phone sender and computer receiver must use the same UDP port.
   * - OSC bind ``ip``
     - ``0.0.0.0``
     - ``--ip`` or real-time Python API
     - Usually keep the default; the phone targets the computer's local IP.
   * - ``label``
     - Required
     - ``--label``
     - Choose the motion class name used to organize recordings.
   * - ``session``
     - UTC timestamp
     - ``--session``
     - Use distinct session names to support session-aware validation.
   * - ``output_dir``
     - ``artifacts/recordings``
     - ``--output-dir``
     - May be any writable local directory.
   * - ``stale_warning``
     - ``0.25`` seconds
     - ``--stale-warning``
     - Diagnostic only; tune for the expected sensor/connection behavior.
   * - ``startup_timeout``
     - ``2`` seconds
     - ``--startup-timeout`` or real-time Python API
     - Stops recording or live inference when all six OSC channels do not
       arrive. This is independent of HTTP inference timing.

For example, only include options that differ from the defaults:

.. code-block:: console

   name-that-move-record \
     --label circle \
     --session participant01_round01 \
     --imu-id 1 \
     --port 10000

Flexible workflow choices
-------------------------

Class labels, number of recording sessions, train/validation session split,
augmentation settings, artifact directory, and model tag are project choices.
Remote-server URL and timeout, TouchDesigner IP/port/OSC base path, and worker
queue sizes are configurable. The live command exposes the backend, URL,
timeout, TouchDesigner destination, model directory, and model tag; the Python
API additionally exposes worker queue sizes. Document these choices so an
experiment or performance can be reproduced.

The remote HTTP timeout controls one server request. The shared OSC startup
timeout controls how long either local or remote live mode waits for initial
sensor data. They solve different failure cases.

Current fixed conventions and limitations
-----------------------------------------

* Model-ready arrays are channel-first: one window is
  ``(n_channels, n_timesteps)`` and a batch is
  ``(n_windows, n_channels, n_timesteps)``.
* The current Movesense OSC adapter expects the six named accelerometer and
  gyroscope paths shown in :doc:`hardware_setup`. Offline preprocessing can use
  another channel configuration, but the OSC adapter must be extended before
  it can receive different paths or additional sensors.
* The reference model uses MiniRocket features plus a linear classifier.
* Direct sensor-to-laptop BLE is not yet included.

Before running inference
------------------------

Check all of the following:

1. Runtime ``IMUWindowConfig`` matches the saved model metadata.
2. Incoming arrays use the expected shape and channel order.
3. Sensor units, scaling, placement, and orientation match training.
4. For OSC, the sender ID, address paths, target computer IP, and UDP port
   match the receiver.

Name That Move validates metadata and array shape early. Physical conventions
such as units and sensor placement remain the user's responsibility because
they cannot be inferred reliably from an array alone.
