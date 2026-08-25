Development and documentation checks
====================================

Run quality checks from the repository root:

.. code-block:: console

   conda activate name-that-move
   python -m pip install ".[dev,docs]"
   python -m pytest
   ruff check .
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m build

Continuous integration
----------------------

GitHub Actions runs installation, tests, Ruff, and package builds on Ubuntu,
macOS, and Windows with Python 3.11. The Ubuntu job additionally builds Sphinx
documentation with warnings treated as errors.

Read the Docs
-------------

The root ``.readthedocs.yaml`` selects Python 3.11, installs the package with
the ``docs`` optional dependencies, and builds ``docs/conf.py``. After the
GitHub repository is imported into Read the Docs, enable pull-request builds to
receive a documentation check and preview for each new PR commit.

Documentation source policy
---------------------------

Narrative guides live under ``docs/``. API pages are generated from NumPy-style
docstrings through Sphinx ``autodoc`` and ``napoleon``. Keep executable tutorial
logic in ``examples/`` and include it with ``literalinclude`` so code and
documentation remain synchronized.
