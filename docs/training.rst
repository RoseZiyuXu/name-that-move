Train the example model
=======================

Example dataset layout
----------------------

The current local example contains three motion classes and three separately
recorded rounds:

.. code-block:: text

   artifacts/datasets/example_data/
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
   python examples/train_example_model.py artifacts/datasets/example_data

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
input metadata. They remain local and Git-ignored until a model is intentionally
selected for public release.
