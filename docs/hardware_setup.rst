Hardware and data flow
======================

Reference setup
---------------

The current tested workflow uses:

* a Movesense Sport six-axis IMU;
* an iPhone running Holonist as the BLE-to-OSC bridge;
* a macOS or Windows computer running Name That Move; and
* a shared Wi-Fi network for the phone and computer.

The data path is:

.. code-block:: text

   Movesense Sport
       │  BLE
       ▼
   iPhone + Holonist
       │  OSC over Wi-Fi
       ▼
   Name That Move on the computer
       ├── recorded IMU windows
       └── optional inference → TouchDesigner or another media system

The phone is the OSC sender. Configure Holonist with the computer's local IP
address and the same UDP port that Name That Move will receive. The default
package port is ``10000``. Both devices must be on the same Wi-Fi network.

Expected OSC addresses
----------------------

For the default Movesense ID ``2``, the receiver expects:

.. code-block:: text

   /m/2/acc/x
   /m/2/acc/y
   /m/2/acc/z
   /m/2/gyro/x
   /m/2/gyro/y
   /m/2/gyro/z

The IMU ID and receiving port are configurable. TouchDesigner can inspect
incoming OSC before recording. Close any application already using the chosen
UDP port before starting the package receiver.

Sampling interpretation
-----------------------

Native sensor measurements and BLE/OSC messages may arrive at non-uniform
times. The real-time pipeline stores the latest value for each named channel
and samples the latest complete six-channel state onto a configurable, uniform
model-input timeline. The default is 48 Hz.

Therefore, 48 Hz is a package configuration chosen for this research workflow,
not a claim that every native sensor message arrives at exactly 48 Hz. Units,
channel order, sampling configuration, and window duration used for inference
must match those used for training.

Alternative hardware
--------------------

Users may replace the Movesense/Holonist bridge with another sensor or data
transport. It must ultimately provide the six named channels and satisfy the
same data contract. Direct Movesense-to-laptop BLE support is future work and
is not required for the current workflow.
