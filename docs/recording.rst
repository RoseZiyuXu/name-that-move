Record labeled IMU windows
==========================

Prepare the connection
----------------------

Before recording:

1. Attach and start the Movesense sensor.
2. Connect the sensor to Holonist over BLE.
3. Connect the phone and computer to the same Wi-Fi network.
4. Configure Holonist to send OSC to the computer's IP and selected port.
5. Confirm that the OSC prefix and all six channel suffixes match the receiver.

Install the real-time option
----------------------------

.. code-block:: console

   conda activate name-that-move
   python -m pip install ".[realtime]"

Start a recording session
-------------------------

.. code-block:: console

   name-that-move-record \
     --label circle \
     --session circle_session_01 \
     --imu-id 1 \
     --port 10000 \
     --sample-rate 48 \
     --window-duration 2

The recorder waits until all six channels have arrived. By default, it stops
with a connection checklist if they do not arrive within two seconds; adjust
this with ``--startup-timeout`` for a slower setup. It then creates
non-overlapping ``(6, 96)`` windows. Press Control-C to stop and flush pending
files.

Use a different OSC namespace
-----------------------------

The default ``--imu-id 1`` expects addresses under ``/m/1``. If another phone
app uses a different leading namespace, replace it without changing the six
channel suffixes:

.. code-block:: console

   name-that-move-record \
     --label circle \
     --session circle_session_01 \
     --imu-id 1 \
     --osc-prefix /wearable/right-wrist \
     --port 10000

This expects ``/wearable/right-wrist/acc/x``, ``acc/y``, ``acc/z``,
``gyro/x``, ``gyro/y``, and ``gyro/z``. The prefix is flexible; these six
suffixes are the current OSC acquisition contract.

Output files
------------

Files are written under:

.. code-block:: text

   artifacts/recordings/<label>/<session>/

Every window produces a ``.pkl`` data file and a matching ``.json`` metadata
file. Metadata include shape, channel order, sampling configuration, OSC
message count, and maximum channel age. The entire ``artifacts/`` directory is
ignored by Git by default.

Connection diagnostics
----------------------

The terminal prints expected OSC paths and reports each completed window. A
stale-channel warning indicates that at least one channel may not have updated
recently. If no windows appear, check BLE connection, phone target IP, Wi-Fi,
OSC port, IMU ID or custom prefix, and address paths.
