Train the example model
=======================

Motion examples and sensor placement
------------------------------------

Wear the Movesense sensor on the **right wrist**. Keep the sensor upright in
the same orientation it would have when you raise your arm to read a watch.
Use the same placement and orientation during recording, training-data
collection, and real-time inference so that the channel directions remain
consistent.

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
----------------------

The public example contains three motion classes and three separately
recorded rounds:

.. code-block:: text

   examples/data/still_triangle_circle/
   ├── still1/       ├── triangle1/       ├── circle1/
   ├── still2/       ├── triangle2/       ├── circle2/
   └── still3/       └── triangle3/       └── circle3/

Each folder contains 30 two-second windows. The suffix identifies the
recording session, not a separate class.

Run the complete workflow
-------------------------

Activate the environment and ensure the latest local source has been installed:

.. code-block:: console

   conda activate name-that-move
   python -m pip install --no-deps .
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

The tutorial includes the actual repository script so that documentation and
executable code stay synchronized:

.. literalinclude:: ../examples/train_example_model.py
   :language: python
   :caption: examples/train_example_model.py
   :linenos:

Saved model files
-----------------

By default, the script creates:

.. code-block:: text

   artifacts/models/example_data_smoke/
   ├── MRF-still_triangle_circle_v0.pt
   ├── MRL-still_triangle_circle_v0.pkl
   └── input_shape-still_triangle_circle_v0.pt

These files contain the MiniRocket feature extractor, exported learner, and
input metadata. Retrained files remain local and Git-ignored so the tutorial
does not overwrite the reviewed public reference model. The ready-to-use model
is stored at ``examples/models/still_triangle_circle/``; its data card and the
dataset data card are available beside the published files in the repository.
