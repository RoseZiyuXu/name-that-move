Name That Move
==============

**Wearable-sensor movement recognition for dance, choreography, and
interactive performance!**

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

**New to Python or terminal commands?** Begin with :doc:`before_you_begin`. It
explains what the terminal is, how to read a command, and how to stop or
correct one safely.

.. container:: feedback-invitation

   **Questions and feedback are welcome**

   Questions, corrections, unsuccessful experiments, and suggestions are all
   valuable contributions. If something in the package or documentation is
   unclear, please `open a GitHub Issue
   <https://github.com/RoseZiyuXu/name-that-move/issues>`_. If GitHub is
   unfamiliar or your message involves private participant data or unpublished
   artistic material, please do not hesitate to `contact Rose
   <https://github.com/RoseZiyuXu>`_. Feedback from artists, researchers,
   students, performers, and first-time command-line users helps this project
   become more useful and welcoming.

Current capabilities
--------------------

Name That Move supports recording, training, saved-model inference, and live
OSC workflows using either a local saved model or a remote HTTP endpoint.
Direct sensor-to-laptop BLE is planned as an optional input route, while OSC
remains the stable and flexible reference workflow.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   before_you_begin
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
