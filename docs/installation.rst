Installation
============

Name That Move supports Python 3.9 or newer. The project and CI currently use
Python 3.11.

Create and activate an environment
----------------------------------

This example uses Conda. Activation must happen **before** installation so the
package is installed into the intended environment rather than ``base``.

.. code-block:: console

   conda create -n name-that-move python=3.11
   conda activate name-that-move
   which python

On Windows, replace ``which python`` with ``where python``. The resulting path
should include the new environment name.

Install from GitHub
-------------------

.. code-block:: console

   python -m pip install "git+https://github.com/RoseZiyuXu/name-that-move.git"

Install a local checkout
------------------------

.. code-block:: console

   git clone https://github.com/RoseZiyuXu/name-that-move.git
   cd name-that-move
   python -m pip install .

This project uses a regular installation rather than editable mode. After
changing package source code, reinstall it before running examples:

.. code-block:: console

   conda activate name-that-move
   python -m pip install --no-deps .

Verify the active installation
------------------------------

.. code-block:: console

   python -c "import name_that_move; print(name_that_move.__file__)"

The printed path should point into the currently active environment.

Developer and documentation tools
---------------------------------

.. code-block:: console

   python -m pip install ".[dev,docs]"
   python -m pytest
   ruff check .
   python -m build

Build this documentation locally with:

.. code-block:: console

   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser to inspect the result.
