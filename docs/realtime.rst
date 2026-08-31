Real-time performance workflow
==============================

Architecture
------------

The pipeline separates time-sensitive acquisition from slower disk and model
operations:

.. code-block:: text

   OSCReceiver → LatestValueWindowBuffer → completed IMU window
                                             ├── AsyncWindowRecorder
                                             ├── InferenceWorker
                                             └── TouchDesigner callback

OSC is the current stable acquisition adapter. Planned direct BLE support will
replace only that first adapter: it will decode sensor packets into the same
named six-channel values and then reuse the existing window buffer and every
downstream component. The BLE option will therefore complement, not replace,
the phone-to-OSC workflow. See :doc:`hardware_setup` for the practical
tradeoffs.

Sampling immediately continues after a window is completed. Saving and
inference use independent bounded worker queues, so disk or network latency
does not pause sampling or create unlimited backlog.

Current status
--------------

The package currently provides:

* a working OSC recording command;
* fixed-rate latest-value window construction;
* asynchronous recording;
* a shared prediction result for local and remote inference;
* a local saved-model predictor and a prototype HTTP model client;
* a fail-fast selector and ``name-that-move-live`` command; and
* a TouchDesigner OSC output adapter.

The live command loads one selected backend, starts sensor acquisition, prints
predictions, and can optionally send them to TouchDesigner. See
:doc:`inference` for local and remote examples. The remote mode still depends
on an independently running, compatible HTTP model server.

Both live modes fail fast if a complete set of six OSC channels does not arrive
within the configurable startup timeout. This prevents an unnoticed connection
or port mismatch from leaving the program waiting indefinitely.

The OSC sender namespace is configurable with ``osc_prefix`` or
``--osc-prefix``. This changes the leading portion of every address while the
six endings—``acc/x``, ``acc/y``, ``acc/z``, ``gyro/x``, ``gyro/y``, and
``gyro/z``—remain fixed so that the window buffer receives an unambiguous
channel order.

Configuration rule
------------------

The real-time :class:`~name_that_move.preprocessing.IMUWindowConfig` must match
the model metadata. A model trained with ``(6, 96)`` windows cannot receive a
different rate, duration, channel order, or unit convention without compatible
preprocessing or retraining.
