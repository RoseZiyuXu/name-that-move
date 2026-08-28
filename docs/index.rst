Name That Move
==============

**IMU motion classification for dance, choreography, and interactive
performance — powered by MiniRocket.**

Name That Move supports a path from six-axis IMU recordings to offline model
training, saved-model inference, and reusable real-time performance components.

This documentation describes the hardware and OSC workflow used by the
project. Other sensors and transports can be used when they produce the same
model-ready data contract.

.. note::

   The package is alpha software. The recording and offline model workflow are
   functional, and live OSC windows can use either a local saved model or a
   remote HTTP endpoint. Direct sensor-to-laptop BLE remains development work.

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
