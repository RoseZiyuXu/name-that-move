Record labeled IMU windows
==========================

Prepare the connection
----------------------

Before recording:

1. Attach and start your six-axis IMU sensor (for example, Movesense Sport).
2. Connect it to your data-transmitter app (for example, Holon.ist over BLE).
3. If the app runs on a phone, connect the phone and computer to the same
   Wi-Fi network.
4. Configure the app to send OSC to the computer's IP and selected port.
5. Confirm that the OSC prefix and all six channel suffixes match the receiver.

Movesense Sport and Holon.ist are the tested reference setup, not package
requirements. You may replace either component as long as the OSC sender
provides the six channels described by the :doc:`data_contract`.

Install OSC support
-------------------

.. code-block:: console

   conda activate name-that-move
   python -m pip install --editable ".[realtime]"

Here, ``--editable`` keeps the environment connected to your local checkout.
``[realtime]`` is the optional dependency group that installs ``python-osc``,
which the recording and live commands need to receive OSC messages. If you
already installed ``.[dev]``, OSC support is included and you can skip this
step.

Start a recording session
-------------------------

.. code-block:: console

   name-that-move-record \
     --label my_move \
     --session my_move_session_01 \
     --imu-id 1 \
     --port 10000 \
     --sample-rate 48 \
     --window-duration 2

Here, ``my_move`` is only a placeholder: replace it with the name of any
movement you want to record. By default, this example writes its files to
``artifacts/recordings/my_move/my_move_session_01/``, relative to the directory
where you run the command. To use another root directory, add
``--output-dir <path>``; the recorder will still create the ``my_move`` and
``my_move_session_01`` subdirectories.

The recorder waits until all six channels have arrived. By default, it stops
with a connection checklist if they do not arrive within two seconds; adjust
this with ``--startup-timeout`` for a slower setup. It then creates
non-overlapping ``(6, 96)`` windows. Press Control-C to stop and flush pending
files.

The terminal prints a new line for every captured window, making it easy to
record toward a target such as 30 windows (you may want to record a few extra
so you can remove the first and last one or two windows and keep a cleaner
middle section). When you stop, it confirms the final window count and save
location:

.. code-block:: text

   Captured window 1
   Captured window 2
   Captured window 3
   ...
   Captured window 34

   Recording stopped.
   Saved 34 windows to: artifacts/recordings/my_move/my_move_session_01

Use a different OSC namespace
-----------------------------

Use ``--osc-prefix`` to customize the beginning of the incoming OSC path. The
default ``--imu-id 1`` expects addresses under ``/m/1``. If another OSC sender
uses a different leading namespace, replace it without changing the six channel
suffixes:

.. code-block:: console

   name-that-move-record \
     --label my_move \
     --session my_move_session_01 \
     --imu-id 1 \
     --osc-prefix /wearable/right-wrist \
     --port 10000

This expects ``/wearable/right-wrist/acc/x``, ``acc/y``, ``acc/z``,
``gyro/x``, ``gyro/y``, and ``gyro/z``. The prefix is flexible; these six
suffixes are the current OSC input format.

Output files
------------

Files are written under:

.. code-block:: text

   artifacts/recordings/<label>/<session>/

Every window produces two files with the same base name:

* The ``.pkl`` file contains the numeric IMU window used for training and
  inference, with shape ``(6, 96)`` under the default configuration.
* The ``.json`` file is a human-readable record of how that window was
  captured. It includes the time, IMU ID, label, session, shape, channel order,
  sampling configuration, OSC message count, and channel-timing diagnostic.

The JSON file does not duplicate the sensor samples. Keep each pair together
so you can identify and inspect the data later without opening the pickle file.
Filenames use the computer's local calendar time followed by two digits for
hundredths of a second, for example, ``imu7_20260901_101955_72.pkl``. If two
windows would still receive the same name, the later pair receives another
suffix such as ``_02`` rather than overwriting the first. The JSON sidecar
retains the exact UTC capture time. The entire ``artifacts/`` directory is
ignored by Git by default.

Connection diagnostics
----------------------

A stale-channel warning appears only when at least one channel has gone longer
than ``--stale-warning`` without an update; the JSON sidecar retains the exact
diagnostic for every window. If no windows appear, check the
sensor-to-transmitter connection, sender target IP, network, OSC port, IMU ID
or custom prefix, and address paths. If stale-channel warnings begin after
recording has started, stop the recorder and restore the stream; do not use
windows captured after that interruption as training data.
