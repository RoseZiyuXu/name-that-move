Train a model
=============

Train with your own recordings
------------------------------

**For every selected motion class, all sessions not named with**
``--validation-session`` **are automatically used for training.** You therefore
choose the held-out sessions explicitly; the command uses the remaining
discovered sessions as the training set.

The training command reads the same folders created by
``name-that-move-record``:

.. code-block:: text

   artifacts/recordings/
   ├── my_move/
   │   ├── my_move_session_01/
   │   └── my_move_session_02/
   └── your_move/
       ├── your_move_session_01/
       └── your_move_session_02/

Use at least two motion classes and record separate sessions for each class.
The following command trains on Session 1 and reserves Session 2 from both
classes for held-out validation:

.. code-block:: console

   conda activate name-that-move
   python -m pip install --editable .
   name-that-move-train \
     --dataset-dir artifacts/recordings \
     --label my_move \
     --label your_move \
     --validation-session my_move_session_02 \
     --validation-session your_move_session_02 \
     --output-dir artifacts/models/our_moves \
     --model-tag our_moves_v0 \
     --epochs 10

``my_move`` and ``your_move`` are placeholders; replace them with your own
recording-label folder names.
Repeat ``--label`` for every class you want to include and
``--validation-session`` for every complete session you want to hold out. You
may hold out more than one session per class by repeating the option; every
other discovered session for that class becomes training data. A session name
is accepted when it is unique; a relative path such as
``my_move/my_move_session_02`` can resolve an ambiguous name.

The command prints the split before training, validates all windows, trains
and saves the model, reloads it, and reports accuracy on the held-out sessions.
A tiny dataset can confirm that the workflow runs, but its accuracy should not
be treated as evidence that the model will generalize to new performers,
sensor placements, or recording conditions.

The example above creates:

.. code-block:: text

   artifacts/models/our_moves/
   ├── MRF-our_moves_v0.pt
   ├── MRL-our_moves_v0.pkl
   └── input_shape-our_moves_v0.pt

Use this directory and tag with ``name-that-move-live`` or
``name-that-move-evaluate`` after training.

Train the bundled example model
-------------------------------

Motion examples and sensor placement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the published example model, wear a six-axis IMU sensor on the **right
wrist**. The example data were recorded with a Movesense Sport, but another
sensor may be used if it provides the same channel contract. Keep the sensor
upright in the same orientation it would have when you raise your arm to read
a watch. Use the same placement and orientation during recording,
training-data collection, and real-time inference so that the channel
directions remain consistent.

The example dataset uses ``still``, ``triangle``, and ``circle``. For
``still``, hold the right wrist comfortably still. The following demonstrations
show the two movement classes. Users are encouraged to wear their sensor in the
same way and imitate these movements when testing the example model in
real time.

.. list-table:: Example movements
   :widths: 50 50
   :header-rows: 1

   * - Triangle
     - Circle
   * - .. image:: _static/triangle.gif
          :alt: Right-wrist demonstration of the triangle movement
          :width: 100%
     - .. image:: _static/circle.gif
          :alt: Right-wrist demonstration of the circle movement
          :width: 100%

Example dataset layout
^^^^^^^^^^^^^^^^^^^^^^

The public example contains three motion classes and three separately
recorded rounds:

.. code-block:: text

   examples/data/still_triangle_circle/
   ├── still1/       ├── triangle1/       ├── circle1/
   ├── still2/       ├── triangle2/       ├── circle2/
   └── still3/       └── triangle3/       └── circle3/

Each folder contains 30 two-second windows. The suffix identifies the
recording session, not a separate class.

Run the training script
^^^^^^^^^^^^^^^^^^^^^^^

Activate the environment and ensure the latest local source has been installed:

.. code-block:: console

   conda activate name-that-move
   python -m pip install --editable .
   python examples/train_example_model.py

The script:

1. validates every window against ``48 Hz × 2 seconds × 6 channels``;
2. groups the folders into ``still``, ``triangle``, and ``circle``;
3. trains on Sessions 1 and 2;
4. holds out Session 3 for cross-session validation;
5. extracts MiniRocket features and trains the linear classifier;
6. saves three model artifacts;
7. reloads those artifacts from disk; and
8. runs inference on the held-out session.

The third command runs the complete example workflow. You do not need to read
or modify its Python code to use the example, but the complete source remains
available here: :download:`examples/train_example_model.py
<../examples/train_example_model.py>`.

Saved model files
^^^^^^^^^^^^^^^^^

By default, the script creates:

.. code-block:: text

   artifacts/models/example_data/
   ├── MRF-still_triangle_circle_v0.pt
   ├── MRL-still_triangle_circle_v0.pkl
   └── input_shape-still_triangle_circle_v0.pt

These files contain the MiniRocket feature extractor, exported learner, and
input metadata. Retrained files remain local and Git-ignored so the tutorial
does not overwrite the reviewed public reference model. The ready-to-use model
is stored at ``examples/models/still_triangle_circle/``; its data card and the
dataset data card are available beside the published files in the repository.
