Offline saved-model inference
=============================

Load the model
--------------

Use the same IMU configuration used during training:

.. code-block:: python

   from name_that_move import IMUWindowConfig, load_model

   config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
   feature_extractor, learner = load_model(
       "examples/models/still_triangle_circle",
       tag="still_triangle_circle_v0",
       expected_config=config,
   )

Predict a recorded window
-------------------------

.. code-block:: python

   from name_that_move import load_segment, predict

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

``load_model`` checks saved shape, sampling rate, duration, and channel order.
``predict`` checks incoming windows against the loaded feature extractor.
Mismatches fail before MiniRocket feature extraction with an actionable error.

The filename above is one window included in the public example dataset.

Evaluate a later recording session
----------------------------------

Use offline session evaluation when the movement has already been recorded.
For example, after training on Sessions 1 and 2, evaluate a new ``circle4``
folder without starting OSC or performing the movement again:

.. code-block:: console

   name-that-move-evaluate \
     --session-dir artifacts/datasets/example_data/circle4 \
     --model-dir examples/models/still_triangle_circle \
     --model-tag still_triangle_circle_v0 \
     --expected-label circle

The command loads every ``.pkl`` window in the folder as one batch and reports
the number of windows, predicted-label distribution, mean confidence, and
accuracy when ``--expected-label`` is supplied. Run it once for each class's
new session folder. Session 4 should remain outside training if the result is
being used as genuinely held-out evidence.

If the model is hosted remotely, replace ``--model-dir`` and ``--model-tag``
with the endpoint. Saved windows are uploaded one at a time; no live OSC input
is needed:

.. code-block:: console

   name-that-move-evaluate \
     --session-dir artifacts/datasets/example_data/circle4 \
     --remote-url https://your-model-server.example/process \
     --http-timeout 2 \
     --expected-label circle

Choose local or remote live inference
-------------------------------------

``name-that-move-live`` connects the same real-time OSC pipeline to exactly
one inference backend. For the local example model:

.. code-block:: console

   name-that-move-live \
     --model-dir examples/models/still_triangle_circle \
     --model-tag still_triangle_circle_v0

For the existing HTTP model server:

.. code-block:: console

   name-that-move-live \
     --remote-url https://your-model-server.example/process \
     --http-timeout 2

The two modes return the same ``Prediction(label, confidence)`` result. The
command requires one mode and rejects attempts to provide both. Local mode
loads the model and verifies its saved IMU configuration before opening the
OSC receiver. Remote mode validates the outgoing window locally; availability
and model compatibility of the external server remain the server operator's
responsibility.

``--startup-timeout`` is different from ``--http-timeout``. The startup
timeout applies to both local and remote modes and stops the pipeline when a
complete set of six OSC channels has not arrived within two seconds. The
remote timeout begins only after a window exists and limits one HTTP request.

Add TouchDesigner output
------------------------

Run the complete public-example command below to receive six-channel IMU input
on UDP port 10000, classify it locally, and forward every successful prediction
to TouchDesigner on UDP port 8000:

.. code-block:: console

   name-that-move-live \
     --model-dir examples/models/still_triangle_circle \
     --model-tag still_triangle_circle_v0 \
     --ip 0.0.0.0 \
     --port 10000 \
     --imu-id 1 \
     --sample-rate 48 \
     --window-duration 2 \
     --startup-timeout 2 \
     --touchdesigner-ip 127.0.0.1 \
     --touchdesigner-port 8000 \
     --touchdesigner-path /sensor/1

TouchDesigner receives the label at ``/sensor/1/label`` and confidence at
``/sensor/1/confidence``. Port 10000 is the sensor/phone-to-package input;
port 8000 is the separate package-to-TouchDesigner output. OSC sampling
continues in its own thread while model inference and downstream output run
through the bounded inference worker.
