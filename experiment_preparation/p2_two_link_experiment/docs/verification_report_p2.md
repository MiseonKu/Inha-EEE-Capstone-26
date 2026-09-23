# P2 Verification Report

Generated: 2026-05-23 Asia/Seoul

## Environment

- Device: Samsung Galaxy Book Ultra 6
- OS mode: Windows local environment
- Python: 3.14.3
- Dependencies observed:
  - numpy 2.4.4
  - pandas 3.0.3
  - matplotlib 3.10.9
  - PyYAML 6.0.3
  - pytest 9.0.3

## Geometry

- `L1_effective_m = 0.2996`
- `L2_effective_m = 0.29945`
- `link1_flexible_bar_length_m = 0.233`
- `link2_flexible_bar_length_m = 0.215`
- `link1_nonflex_offset_mm = 66.6`
- `link2_nonflex_offset_mm = 84.45`
- material preset: `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`

## Status Matrix

| Check | Status | Evidence |
|---|---|---|
| STEP file copied | PASS | `source_step/2_LINK_SO101_ASSEMBLY.step` exists |
| CAD STEP measurement | PASS_WITH_WARNING | FreeCAD/CadQuery unavailable; measurement skipped |
| Mesh conversion | NOT_TESTED | Not required; STEP is visual reference only |
| FK check | PASS | 4 validation cases passed |
| Target generation | PASS | 50 targets, 20000 candidates, 19722 valid candidates, 105 occupied bins |
| Bin balance | PASS | selected bin min 1, max 1, max-minus-min 0 |
| Trial schedule | PASS | 150 trials, 50 targets, 3 repeats, adjacent same target count 0 |
| Trajectory generation | PASS | 41850 rows, 150 trials |
| Matplotlib plots | PASS | 7 PNG plots generated under `plots/pre_imu_p2/` |
| Dataset validation with real Vive data | NOT_TESTED | No real P2 Vive dataset provided |
| Pytest | PASS | `7 passed in 0.56s` |
| Hardware command safety | PASS | No firmware, serial, torque, INIT, calibration, MOVE, or motor command used |

## Commands Run

```powershell
python .\scripts\check_p2_fk.py
python .\scripts\inspect_p2_step.py
python .\scripts\generate_p2_forward_targets.py --config .\config\p2_forward_workspace.yaml --target-count 50 --output .\outputs\pre_imu_p2\p2_targets_pilot.csv --metadata-output .\outputs\pre_imu_p2\p2_targets_pilot_metadata.json
python .\scripts\generate_p2_trial_schedule.py --targets .\outputs\pre_imu_p2\p2_targets_pilot.csv --repeats 3 --output .\outputs\pre_imu_p2\p2_trial_schedule.csv --metadata-output .\outputs\pre_imu_p2\p2_trial_schedule_metadata.json
python .\scripts\generate_p2_trajectories.py --schedule .\outputs\pre_imu_p2\p2_trial_schedule.csv --output .\outputs\pre_imu_p2\p2_trajectories.csv
python .\scripts\plot_p2_workspace.py --targets .\outputs\pre_imu_p2\p2_targets_pilot.csv --trajectories .\outputs\pre_imu_p2\p2_trajectories.csv --output-dir .\plots\pre_imu_p2
python .\scripts\validate_p2_dataset.py --report .\outputs\pre_imu_p2\p2_dataset_validation_report.md
python -m pytest .\tests -q
```

## Files Created

- config YAML files
- FK, target, schedule, trajectory modules
- CLI scripts
- docs and this verification report
- copied STEP file
- target/schedule/trajectory CSV outputs
- metadata JSON outputs
- matplotlib PNG plots
- tests

## Issues Found And Fixed

- Initial test for forbidden IMU/Vive columns treated `azimuth` as containing `imu`. The test was corrected to check actual sensor-column prefixes and tracker labels.
- Added missing `p2_workspace_preview.csv` output.
- Added trajectory metadata JSON.

## Remaining Issues

- STEP dimensions were not measured because CAD tooling is unavailable.
- Real Vive-to-base calibration is not available yet.
- Real settled dataset validation is not tested until P2 experiment logs exist.
- Motor sign, encoder zero, and radian scale must be verified before hardware use.
