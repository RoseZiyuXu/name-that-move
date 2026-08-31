Hardware and data flow
======================

Supported OSC setup
-------------------

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

Acquisition choices
-------------------

Name That Move keeps sensor acquisition separate from window construction,
recording, and inference. The stable release currently supports the
phone-to-OSC path above. A direct Movesense-to-laptop BLE adapter is under
development and will be offered as an additional choice rather than replacing
OSC.

.. list-table:: OSC bridge and direct BLE at a glance
   :header-rows: 1
   :widths: 18 38 38

   * - Transport
     - Advantages
     - Tradeoffs
   * - Phone + OSC
     - Uses the tested performance workflow; keeps the Bluetooth connection
       close to a performer; Wi-Fi can cover a larger stage-to-computer
       distance; and works with tools such as TouchDesigner.
     - Requires a phone, a compatible bridge app, shared Wi-Fi, and matching
       IP address, port, IMU ID, and OSC paths.
   * - Direct BLE
     - Removes the phone and Wi-Fi bridge; reduces setup steps; and lets the
       package receive sensor measurements directly on the laptop.
     - Requires compatible sensor firmware and a laptop BLE connection; has a
       shorter practical range; and is not yet part of the stable release.

Both choices will converge on the same six-channel sample interface and the
same :doc:`data_contract`. The selected transport must not change channel
names, order, units, sampling configuration, or model compatibility. After
acquisition, both paths reuse the same window buffer, background recorder,
local or remote inference worker, and optional TouchDesigner output.

.. code-block:: text

   Movesense ── BLE ──> phone + OSC ──┐
                                      ├──> shared IMU samples and windows
   Movesense ── direct BLE ───────────┘       ├── recording
                                              ├── local/remote inference
                                              └── TouchDesigner output

Expected OSC addresses
----------------------

For the default Movesense ID ``1``, the receiver expects:

.. code-block:: text

   /m/1/acc/x
   /m/1/acc/y
   /m/1/acc/z
   /m/1/gyro/x
   /m/1/gyro/y
   /m/1/gyro/z

The IMU ID and receiving port are configurable. Another sender app may replace
the leading ``/m/1`` namespace with ``--osc-prefix``; for example,
``--osc-prefix /wearable/right-wrist`` produces
``/wearable/right-wrist/acc/x`` through
``/wearable/right-wrist/gyro/z``. The six suffixes remain fixed. TouchDesigner
can inspect incoming OSC before recording. Close any application already using
the chosen UDP port before starting the package receiver.

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
same data contract. Direct Movesense-to-laptop BLE support is optional future
work and is not required for the current OSC workflow.
