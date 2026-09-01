Run inference
=============

After training a model, use inference to turn new IMU windows into movement
labels and confidence values. Name That Move gives you two independent
choices:

* **When the movement data are processed:** use a completed recording session,
  or classify new movement live as sensor data arrive.
* **Where the model runs:** load saved model files on your computer, or send
  each window to a compatible remote HTTP model server, such as one hosted in
  the cloud.

Together, these choices produce four modes:

.. list-table:: Inference modes
   :class: inference-modes-table
   :widths: 24 38 38
   :header-rows: 1

   * - Movement data
     - Local model
     - Remote model
   * - Recorded session
     - Mode 1
     - Mode 2
   * - Live sensor stream
     - Mode 3
     - Mode 4

The live modes are especially useful in performance and interactive-media
contexts: a performer can make a new movement and receive its label and
confidence while the movement is happening. All four modes use the same IMU
window contract and produce the same class meaning, provided that the selected
model was trained with matching sensor configuration, units, placement, and
orientation.

1. Recorded session + local model
---------------------------------

Use this mode after a movement session has been recorded and the saved model
files are available on the same computer. No live OSC stream or network
connection is needed.

This public example evaluates the recorded ``circle3`` session with the
bundled model:

.. code-block:: console

   name-that-move-evaluate \
     --session-dir examples/data/still_triangle_circle/circle3 \
     --model-dir examples/models/still_triangle_circle \
     --model-tag still_triangle_circle_v0 \
     --expected-label circle

To evaluate your own later recording, replace ``--session-dir``,
``--model-dir``, ``--model-tag``, and ``--expected-label`` with your paths and
label. Keep the evaluated session outside training if you intend to describe
the result as held-out evidence.

2. Recorded session + remote model
----------------------------------

Use this mode when the movement is already recorded but inference should run
on a compatible remote HTTP model server. The command reads each saved
``.pkl`` window and uploads it to the endpoint; no live OSC input is needed.
Name That Move does not currently publish a hosted model server, so this is a
configuration template rather than a runnable public example:

.. code-block:: console

   name-that-move-evaluate \
     --session-dir examples/data/still_triangle_circle/circle3 \
     --remote-url https://my-model.example/process \
     --http-timeout 2 \
     --expected-label circle

The ``.example`` URL is intentionally a placeholder. Replace it with an
available server that implements the expected Name That Move request and
response format.

3. Live sensor stream + local model
-----------------------------------

Use this mode to classify new movement in real time with model files stored on
the performance computer. The command listens for live six-channel OSC data,
constructs complete windows, and prints each predicted label and confidence.

.. code-block:: console

   name-that-move-live \
     --model-dir examples/models/still_triangle_circle \
     --model-tag still_triangle_circle_v0 \
     --ip 0.0.0.0 \
     --port 10000 \
     --imu-id 1

If the incoming OSC addresses do not begin with the default ``/m/1``, add
``--osc-prefix /your/prefix`` to replace that leading path. The six channel
endings, from ``acc/x`` through ``gyro/z``, remain fixed. See
:doc:`configuration` for more examples.

4. Live sensor stream + remote model
------------------------------------

Use this mode to receive live sensor data on the performance computer while a
compatible HTTP server performs inference. Completed windows are sent without
blocking the real-time sampling thread.

.. code-block:: console

   name-that-move-live \
     --remote-url https://my-model.example/process \
     --http-timeout 2 \
     --ip 0.0.0.0 \
     --port 10000 \
     --imu-id 1

The local and remote live modes return the same ``Prediction(label,
confidence)`` result. ``name-that-move-live`` requires exactly one backend and
rejects attempts to provide both ``--model-dir`` and ``--remote-url``.

Understand the common settings and results
------------------------------------------

For recorded-session inference, ``name-that-move-evaluate`` loads every
``.pkl`` window in the selected folder and reports the window count,
predicted-label distribution, mean confidence, and accuracy when
``--expected-label`` is supplied.

For live inference, ``--startup-timeout`` controls how long the pipeline waits
for all six OSC channels to arrive. It is different from ``--http-timeout``,
which begins only after a complete window exists and limits one remote HTTP
request. If the OSC sender uses another namespace, add ``--osc-prefix`` while
preserving the six ``acc/x`` through ``gyro/z`` suffixes.

Local mode loads the model and checks its saved shape, sampling rate, duration,
and channel order. Remote mode validates outgoing windows locally, but server
availability and model compatibility remain the server operator's
responsibility.

Predict one recorded window with Python
---------------------------------------

For a custom application or notebook, load a local model and predict one saved
window through the Python API:

.. code-block:: python

   from name_that_move import (
       IMUWindowConfig,
       load_model,
       load_segment,
       predict,
   )

   config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
   feature_extractor, learner = load_model(
       "examples/models/still_triangle_circle",
       tag="still_triangle_circle_v0",
       expected_config=config,
   )
   window = load_segment(
       "examples/data/still_triangle_circle/circle3/"
       "imu9_20260824_193911_406.pkl"
   )
   probabilities, labels = predict(
       window,
       feature_extractor,
       learner,
       config=config,
   )
   print(labels[0])
   print(probabilities[0])

``load_model`` and ``predict`` reject incompatible metadata or window shapes
before MiniRocket feature extraction. To connect live predictions to
TouchDesigner or another creative system, continue to :doc:`realtime`.
