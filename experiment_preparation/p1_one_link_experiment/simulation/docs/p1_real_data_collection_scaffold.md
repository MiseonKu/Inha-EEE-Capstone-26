# P1 Real Data Collection Scaffold

This project now has simulation-side files plus scaffolding for the later real
robot/Vive workflow. It still does not send hardware commands.

## One-Command Simulation Demo

From the project root:

```bash
scripts/p1_setup_local_deps.sh
source /opt/ros/$ROS_DISTRO/setup.bash
scripts/p1_run_demo.sh --target-count 80
```

This does the following:

1. Generate `N` balanced P1 targets.
2. Generate a balanced schedule.
3. Generate smooth-return trajectories:
   `basic -> target -> hold -> basic`.
4. Build the ROS2 workspace.
5. Launch RViz2 with target markers and the rigid reference motion.

Prepare files without RViz:

```bash
scripts/p1_prepare_only.sh --target-count 120 --repeats 3
```

## Balanced Target Count

`--target-count N` can be any positive integer. The sampler:

- samples valid forward-hemisphere candidates,
- groups them by `yaw_bin|pitch_bin|height_bin`,
- allocates nearly equal target counts over occupied bins,
- then chooses spread-out points inside each bin.

This avoids center-heavy/Gaussian-like point concentration and keeps target
probability close to constant over the selected workspace bins.

## Vive Comparison Pipeline

The intended live comparison is:

```text
p_target_base          fixed target from CSV
q_encoder              actual robot encoder angle, published as JointState
p_vive_world           Vive Tracker PoseStamped
p_vive_base            calibrated Vive point in base_link
error                  p_vive_base - p_target_base
```

Required calibration:

```bash
python3 scripts/calibrate_p1_vive_to_base.py \
  --pairs outputs/pre_imu_p1/p1_vive_calibration_pairs.csv \
  --output outputs/pre_imu_p1/p1_vive_to_base_calibration.json
```

The pair CSV must contain:

```text
p_target_base_x_m,p_target_base_y_m,p_target_base_z_m,
p_vive_world_x_m,p_vive_world_y_m,p_vive_world_z_m
```

Live comparator node:

```bash
ros2 launch alphaf_p1_sim p1_experiment_setup.launch.py \
  use_rviz:=true \
  joint_source:=trajectory \
  use_vive_comparator:=true \
  vive_pose_topic:=/vive_tracker/pose \
  calibration_json:=$PWD/../../outputs/pre_imu_p1/p1_vive_to_base_calibration.json \
  comparison_log:=$PWD/../../outputs/pre_imu_p1/p1_live_comparison_log.csv
```

It publishes:

- `/p1_vive_comparison`: target marker, Vive marker, and error line.
- optional CSV log with target, Vive, encoder, and error columns.

Build settled ML dataset rows:

```bash
python3 scripts/build_p1_settled_dataset.py \
  --log outputs/pre_imu_p1/p1_live_comparison_log.csv \
  --output outputs/pre_imu_p1/p1_settled_dataset.csv \
  --settle-window-s 0.5
```

## What Is Still Hardware-Specific

The remaining hardware integration must provide:

- a motor command node that safely sends q commands,
- an encoder reader publishing actual `sensor_msgs/JointState`,
- a Vive Tracker driver publishing `geometry_msgs/PoseStamped`,
- a session controller that starts/stops trials and enforces safety.

Those are intentionally not implemented as real hardware commands here.
