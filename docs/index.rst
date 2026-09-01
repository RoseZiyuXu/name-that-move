Name That Move
==============

**IMU-based motion classification for dance, choreography, and interactive
performance — powered by MiniRocket.**

Name That Move supports a path from six-axis IMU recordings to offline model
training, saved-model inference, and reusable real-time performance components.
The tutorials follow that path from recording your own movement data, through
training and inference, to building a creative application with the results.

`MiniRocket <https://arxiv.org/abs/2012.08791>`_—short for MINImally RandOm
Convolutional KErnel Transform—is the time-series feature transform used by
the example training pipeline.

This documentation describes the hardware and OSC workflow used by the
project. Other sensors and transports can be used when they produce the same
model-ready data contract.

.. note::

   Recording, training, and saved-model inference are functional. Live OSC
   windows can use either a local saved model or a remote HTTP endpoint. Direct
   sensor-to-laptop BLE remains development work.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   installation
   hardware_setup
   data_contract
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   recording
   training
   inference
   realtime

.. toctree::
   :maxdepth: 2
   :caption: Reference

   citing
   contributors
   api
   development

.. toctree::
   :maxdepth: 2
   :caption: Contributing and future work

   contributing
   roadmap
