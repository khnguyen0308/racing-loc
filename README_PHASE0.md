# Phase 0 tools — Camera localization, Yas Marina (Constructor Labs challenge)

Team scripts that run **on top of the organisers' starter kit** (`tools/extract_gt.py`, `tools/evaluate.py`).
They turn the 7 GiB mcap into small files everybody can use, split the run into laps, and produce
the first scored baseline (B0, dead reckoning). Steps 1-8 have been run on the real file (2026-10-16);
the B0/EKF commands and numbers below were re-checked on 2026-10-17.

## Setup (Windows, PowerShell)
Use the project venv `.racing-loc`. **Do not call `python3`**: on this machine it resolves to the
MSYS2 Python (`C:\msys64\ucrt64\bin\python3.exe`), which has none of the packages (`No module named 'mcap'`).
Run every command from the repo root (scripts use relative paths `data/`, `out/`, `config/`).

```powershell
.\.racing-loc\Scripts\Activate.ps1          # afterwards `python` = the venv
pip install mcap mcap-ros2-support numpy pandas scipy matplotlib opencv-python pyyaml
New-Item -ItemType Directory -Force out     # scripts do not create out/ themselves
$M = "data/c0_2026-10-13_22-53-42_camloc_0.mcap"
```
Linux/macOS: `source .racing-loc/bin/activate`, `M=data/...mcap`, `mkdir -p out`, then the same
commands with `$M` and `\` line continuation instead of PowerShell's backtick.

## Pipeline
```powershell
# 1. what is inside (exact field names of state_estimation!)
python tools_team/inspect_mcap.py $M | Out-File -Encoding utf8 notes/mcap_inspect.txt

# 2. ground truth (organisers' script)
python tools/extract_gt.py $M data/gt.csv

# 3. calibration + extrinsics + track borders -> data/static.json
python tools_team/extract_static.py $M data/static.json

# 4. car state, ALL numeric fields -> allowed columns: config/state_columns.yaml
python tools_team/extract_state.py $M data/state_full.csv

# 5. image index (+ every 10th JPEG of 4 cameras for development) and camera timing
python tools_team/extract_images.py $M data/images --cams fl fr cl cr --save-every 10
python tools_team/check_sync.py data/images/index.csv

# 6. centre line, laps, plot  (-> data/laps.csv, data/laps_centre.npy, out/track_gt.png)
New-Item -ItemType Directory -Force out
python tools_team/track_and_laps.py data/static.json data/gt.csv data/laps.csv out/track_gt.png

# 7. validation GT for the 2 held-out laps (the run has 7 full laps -> validation = 6 and 7)
python tools_team/crop_gt.py data/gt.csv data/laps.csv 6 7 --out data/gt_val.csv
# 7b. car-body masks (manual polygons + data-driven check) -> data/masks/{cam}_valid.png
python tools_team/masks.py --images data/images --out data/masks --auto --state data/state_full.csv

# 8. baseline B0 = first pose + speed + yaw rate
python tools_team/b0_dead_reckoning.py --state data/state_full.csv --gt data/gt_val.csv `
    --stamps data/images/index.csv --cam fl --speed-col v_mps --yawrate-col wz_radps `
    --out out/b0_val.csv
python tools/evaluate.py data/gt_val.csv out/b0_val.csv
#    + yaw-rate bias estimated on training laps 1-5 (value is subtracted)
python tools_team/b0_dead_reckoning.py --state data/state_full.csv --gt data/gt_val.csv `
    --stamps data/images/index.csv --cam fl --speed-col v_mps --yawrate-col wz_radps `
    --yawrate-bias -0.00023 --out out/b0_val_bias.csv
python tools/evaluate.py data/gt_val.csv out/b0_val_bias.csv
#    full run from the first pose: use --gt data/gt.csv and evaluate against data/gt.csv
```

## Rules we must not break
- `state_estimation` contains GT (position, orientation, s, n). Only speed, accelerations, yaw rate
  (and similar body-frame motion fields) may be used. If a velocity is expressed in the `map` frame it
  encodes the heading -> use only its magnitude.
- Load the car state with `tools_team/state_io.load_state()`: it raises on any column that is not in
  `allowed` of `config/state_columns.yaml`.
- GT at inference only for the first pose. Causal: pose at stamp t uses data with stamps <= t.
- Never train or tune on the validation laps (6, 7).

## Fake-data check (what "correct" looks like)
`python tests/make_fake_mcap.py data/fake.mcap` writes a small mcap with the same topics; the
pipeline above runs on it unchanged (`$M = "data/fake.mcap"`).

| input | position RMSE | lateral RMSE | yaw RMSE |
|---|---|---|---|
| exact speed + yaw rate | 0.32 m | 0.20 m | 0.04 deg |
| speed +1 %, yaw-rate bias 0.002 rad/s | 28.4 m | 11.0 m | 4.2 deg |

One lap of 64 s is enough for small sensor errors to grow to tens of metres: dead reckoning is the
motion model, the cameras must provide the corrections.

## Real data, first results
- Cameras fl/fr/cl/cr: 11944 frames each, 10.00 Hz, stamps identical within 0.02 ms (hardware sync);
  transport delay log_time - stamp: median 42 ms.
- Car state: `vx_mps` / `v_mps` are body frame (corr 0.9994 with GT), `wz_radps` is the yaw rate
  (corr 0.99976, bias ≈ -0.00023 rad/s on training laps 1-5).
- Track: centre line 3005 m, width 11-20 m. 7 full laps; validation = laps 6+7 (223.5 s, 2234 frames).

| B0 dead reckoning (`v_mps` + `wz_radps`) | position RMSE | lateral | longitudinal | yaw |
|---|---|---|---|---|
| validation laps 6-7 | 19.94 m | 14.86 m | 13.29 m | 2.51 deg |
| + `--yawrate-bias -0.00023` | 7.84 m | 5.40 m | 5.69 m | 0.83 deg |
| full run from the first pose (1194 s) | 65.41 m | 50.47 m | 41.61 m | 8.64 deg |

The bias row is very sensitive: -0.000226 -> 8.04 m, -0.000245 -> 7.10 m. Do not tune it on laps 6-7;
the EKF (step 6) estimates the bias online instead.

## Step 5 - calibration check / refinement (P1) — archived
Method 2 was rejected (see [archive/method2/README.md](archive/method2/README.md)); the refine result was wrong and is not used.
Needs the JPEGs from pipeline step 5; uses GT (development only).
```powershell
python archive/method2/project_borders.py --cam cr --n 12 --out out/overlay_cr            # look at the overlays
python archive/method2/project_borders.py --cam cr --n 40 --refine --out out/overlay_cr   # -> config/extrinsic_correction.json
```
Tested by drawing borders with a known error (yaw +1.5, pitch -0.7, roll +0.4 deg) on synthetic
images: recovered +1.4 / -0.5 / +0.5 deg. base_link is ~at road level (GT z - border z = -0.07 m),
so the cameras are only ~0.4 m above the asphalt.

## Step 6 - EKF (P2)
```powershell
python loc/run_ekf.py --state data/state_full.csv --gt data/gt_val.csv --stamps data/images/index.csv --out out/v1_val.csv
python tools/evaluate.py data/gt_val.csv out/v1_val.csv
python loc/whatif_measurements.py --state data/state_full.csv --gt data/gt_val.csv --stamps data/images/index.csv
```
`run_ekf.py` also writes `out/v1_val_debug.csv` (bias, speed scale, sigma). Without `--meas` it is B0
(20.01 m). `whatif_measurements.py` uses GT to simulate camera measurements -> planning only, never a submission.

| simulated camera measurement (validation laps 6-7) | pos RMSE | lat RMSE | yaw RMSE | <1 m |
|---|---|---|---|---|
| none (= B0) | 20.01 | 14.89 | 2.51 deg | 1.5 % |
| lateral 0.3 m + yaw 0.5 deg @10 Hz, no along-track info | 2.08 | 0.16 | 0.14 | 14.8 % |
| + along-track 1 m in corners (22 % of frames) | 1.47 | 0.16 | 0.14 | 51.1 % |
| same @2 Hz | 1.79 | 0.28 | 0.21 | 32.2 % |
| lateral 0.6 m | 1.45 | 0.24 | 0.14 | 51.2 % |
| lateral 0.15 m, along 0.5 m in corners | 1.15 | 0.16 | 0.15 | 66.6 % |
| 10 % outliers, gating ON | 1.47 | 0.16 | 0.14 | 50.7 % |
| 10 % outliers, gating OFF | 1.96 | 0.72 | 0.28 | 33.7 % |
