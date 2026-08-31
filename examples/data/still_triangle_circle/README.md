# Still-Triangle-Circle example dataset

This deliberately public dataset accompanies the Name That Move tutorial. It
contains three movement classes recorded by Ziyu "Rose" Xu with one Movesense
Sport sensor worn on the right wrist in an upright, watch-like orientation.

## Contents

- Classes: `still`, `triangle`, and `circle`
- Recording sessions: three per class
- Windows: 30 per class and session; 270 total
- Window duration: 2 seconds
- Tutorial sampling timeline: 48 Hz
- Channels: accelerometer x/y/z followed by gyroscope x/y/z
- Stored array shape: `(1, 6, 96)`; the loader normalizes this to `(6, 96)`

Folders ending in `1`, `2`, and `3` are separate recording sessions, not
additional movement classes. The tutorial trains on Sessions 1-2 and holds out
Session 3 for cross-session validation. This demonstrates the package workflow
and transfer to a later recording by the same performer with the same sensor;
it is not a cross-performer, cross-device, or general accuracy benchmark.

The numeric values preserve the recording pipeline's six IMU channels. Do not
assume physical units unless your acquisition system documents and verifies
them. Training and inference must use the same channel order, sensor placement,
orientation, sampling timeline, and window duration.

Python pickle files can execute code while loading. Load these files only from
this trusted repository, and do not load untrusted `.pkl` files.

Unless otherwise noted, these example files are provided under the repository's
[MIT License](../../../LICENSE).

