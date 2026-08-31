# TouchDesigner downstream example

[`name_that_move_visualizer.toe`](name_that_move_visualizer.toe) is a small
creative-coding example for the public `still` / `triangle` / `circle` model.
It receives Name That Move predictions over OSC, smooths and thresholds the
confidence signal, and maps accepted labels to simple shape visuals.

The patch expects:

- `/sensor/1/label` on UDP port `8000`
- `/sensor/1/confidence` on UDP port `8000`

From the repository root, open the patch in TouchDesigner and run:

```bash
name-that-move-live \
  --model-dir examples/models/still_triangle_circle \
  --model-tag still_triangle_circle_v0 \
  --ip 0.0.0.0 \
  --port 10000 \
  --imu-id 1 \
  --sample-rate 48 \
  --window-duration 2 \
  --startup-timeout 2 \
  --touchdesigner-ip 127.0.0.1 \
  --touchdesigner-port 8000 \
  --touchdesigner-path /sensor/1
```

Port `10000` receives IMU data from the phone or sensor bridge. Port `8000`
separately sends classification results from Name That Move to TouchDesigner.
Only one application can listen on a given UDP port at a time.

Use this patch as a starting point: replace the triangle and circle visuals
with images, sound, lighting, animation, or other performance controls. It was
created and tested with TouchDesigner `2025.33070`.
