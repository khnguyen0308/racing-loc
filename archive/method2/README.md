# Archived: INTRO method 2 (track-border matching) — tried, rejected

Kept for the failure analysis. Not used by `run.py`.

| file | what |
|---|---|
| `project_borders.py` | project the map borders into the images at the GT pose; optional grid "refine" of camera yaw/pitch/roll |
| `border_meas.py` | render-and-compare measurement: shift the predicted pose across the track, keep the shift whose projected borders lie on image edges (Canny / top-hat) |
| `test_border_meas.py` | dev check: map z / pitch / roll vs GT (`attitude`), pull-back from a 1 m offset (`recover`) |
| `run_cam.py` | EKF with these lateral measurements |

What we measured (training laps 1-5):

- Map-derived road z / pitch / roll is good: z within ~5 cm, pitch p95 0.5° (vs 1.9° with pitch = 0). This part
  was reused in `loc/vision.py` (`MapGround`).
- One measurement at the GT pose: only 23-38 % of frames within 0.3 m, IQR ±0.7-1 m, `cl` biased by ~-1 m —
  even with the GT pitch/roll. Reason: 6-8 parallel bright lines at the track edge (paint, kerb, barrier layers)
  10-20 px apart, while 0.5 m of lateral shift moves a border only ~5 px for a camera 0.35 m above the road.
- In the EKF it made things worse: 51.8 m vs 41.3 m without camera; a wrong accepted fix corrupted the heading
  through the filter correlations.
- `--refine` of the camera angles on edge cost was unreliable (fl -4.6° yaw put the borders into the sky, cl
  +6.6°): the cost can be lowered by pushing border points out of the image, and yaw is nearly unobservable
  for side cameras. The organisers' README later confirmed cl/cr carry an older calibration.

Run from the repository root, e.g. `python archive/method2/test_border_meas.py attitude`.
