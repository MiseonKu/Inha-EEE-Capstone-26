# AlphaF P2 Two-Link Experiment Preparation

이 패키지는 `P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL` 실험을 위한 Windows/Python/matplotlib 기반 준비 코드입니다. ROS2, Gazebo, RViz, 실제 모터 명령은 포함하지 않습니다.

## P2 실험 정의

- q1: base yaw
- q2: shoulder pitch
- q3: elbow pitch
- link 1과 link 2 모두 장착
- Vive Tracker는 link2 tip / `tool0`에 장착 예정
- IMU는 이 단계에서 사용하지 않음
- target은 rigid FK/IK reference coordinate
- label은 Vive Tracker settled measured position
- Vive Tracker는 control feedback이 아니라 ground-truth label/evaluation 전용

## Geometry Authority

STEP 파일은 visual/geometry reference입니다. FK 길이, joint axis, pitch sign, yaw sign의 authority가 아닙니다.

Authoritative values:

- `L1_effective_m = 0.2996`
- `L2_effective_m = 0.29945`
- `link1_flexible_bar_length_m = 0.233`
- `link2_flexible_bar_length_m = 0.215`
- `link1_nonflex_offset_mm = 66.6`
- `link2_nonflex_offset_mm = 84.45`
- material preset: `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`

Effective length is the kinematic motor-axis/tool distance. Flexible bar length is the printed TPU bar length. The offset records adapters and non-flexible geometry and must not be hidden.

## Setup

```powershell
cd 06_simulation_and_data_collect_preparation/p2_two_link_experiment/p2_simulation_matplot
python -m pip install -r requirements.txt
```

## Run Pipeline

```powershell
python .\scripts\check_p2_fk.py
python .\scripts\inspect_p2_step.py

python .\scripts\generate_p2_forward_targets.py `
  --config .\config\p2_forward_workspace.yaml `
  --target-count 50 `
  --output .\outputs\pre_imu_p2\p2_targets_pilot.csv `
  --metadata-output .\outputs\pre_imu_p2\p2_targets_pilot_metadata.json

python .\scripts\generate_p2_trial_schedule.py `
  --targets .\outputs\pre_imu_p2\p2_targets_pilot.csv `
  --repeats 3 `
  --output .\outputs\pre_imu_p2\p2_trial_schedule.csv `
  --metadata-output .\outputs\pre_imu_p2\p2_trial_schedule_metadata.json

python .\scripts\generate_p2_trajectories.py `
  --schedule .\outputs\pre_imu_p2\p2_trial_schedule.csv `
  --output .\outputs\pre_imu_p2\p2_trajectories.csv

python .\scripts\plot_p2_workspace.py `
  --targets .\outputs\pre_imu_p2\p2_targets_pilot.csv `
  --trajectories .\outputs\pre_imu_p2\p2_trajectories.csv `
  --output-dir .\plots\pre_imu_p2

python -m pytest .\tests -q
```

## Outputs

- `outputs/pre_imu_p2/p2_targets_pilot.csv`
- `outputs/pre_imu_p2/p2_targets_pilot_metadata.json`
- `outputs/pre_imu_p2/p2_trial_schedule.csv`
- `outputs/pre_imu_p2/p2_trial_schedule_metadata.json`
- `outputs/pre_imu_p2/p2_trajectories.csv`
- `outputs/pre_imu_p2/p2_trajectories_metadata.json`
- `outputs/pre_imu_p2/p2_workspace_preview.csv`
- `plots/pre_imu_p2/*.png`

## Future Vive Validation

Future real logs should preserve `target_id`, `trial_id`, q command/encoder values, target coordinates, geometry metadata, and Vive settled label columns. `scripts/calibrate_p2_vive_to_base.py`, `scripts/build_p2_settled_dataset.py`, and `scripts/validate_p2_dataset.py` are scaffolds for this later phase.

## Safety

This package never sends motor commands. It does not upload firmware, open serial monitors, run calibration, INIT, torque, or MOVE commands.
