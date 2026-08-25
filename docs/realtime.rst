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

Sampling immediately continues after a window is completed. Saving and
inference use independent bounded worker queues, so disk or network latency
does not pause sampling or create unlimited backlog.

Current status
--------------

The package currently provides:

* a working OSC recording command;
* fixed-rate latest-value window construction;
* asynchronous recording;
* a reusable local or remote inference worker;
* a prototype HTTP model client; and
* a TouchDesigner OSC output adapter.

A single command that loads the local example model, starts sensor acquisition,
and sends predictions to TouchDesigner is still planned. Until that integration
is completed, use the public components in :doc:`api` or the established
Windows prototype workflow.

Configuration rule
------------------

The real-time :class:`~name_that_move.preprocessing.IMUWindowConfig` must match
the model metadata. A model trained with ``(6, 96)`` windows cannot receive a
different rate, duration, channel order, or unit convention without compatible
preprocessing or retraining.
