Offline saved-model inference
=============================

Load the model
--------------

Use the same IMU configuration used during training:

.. code-block:: python

   from name_that_move import IMUWindowConfig, load_model

   config = IMUWindowConfig(sample_rate_hz=48, window_duration_s=2)
   feature_extractor, learner = load_model(
       "artifacts/models/example_data_smoke",
       tag="still_triangle_circle_v0",
       expected_config=config,
   )

Predict a recorded window
-------------------------

.. code-block:: python

   from name_that_move import load_segment, predict

   window = load_segment(
       "artifacts/datasets/example_data/circle3/example_window.pkl"
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

The exact filename above is a placeholder. Replace it with a real ``.pkl``
window present in the selected dataset folder.
