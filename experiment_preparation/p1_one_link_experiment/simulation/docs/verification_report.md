# P1 Verification Report

Verification run: `2026-05-20T01:10:12+09:00` on local Ubuntu.

## Environment

- OS: Ubuntu 26.04 LTS `resolute`
- Kernel: `7.0.0-15-generic`
- Python: `3.14.4`
- ROS_DISTRO: `lyrical`
- `ros2`: `/opt/ros/lyrical/bin/ros2`

ROS/Gazebo package availability:

- `gazebo_ros`: no
- `ros_gz_sim`: yes, `/opt/ros/lyrical`
- `robot_state_publisher`: yes, `/opt/ros/lyrical`
- `xacro`: yes, `/opt/ros/lyrical`
- `rviz2`: yes, `/opt/ros/lyrical`
- `joint_state_publisher`: no
- `joint_state_publisher_gui`: no

## Sources

- External SO-ARM100/SO-101 source: <https://github.com/TheRobotStudio/SO-ARM100>
- Inspected commit: `fda892cba81032c46c40976a48c9ceadbf40a9ca`
- License: Apache-2.0
- Upstream files copied: none
- Upstream files referenced: `Simulation/README.md`, `Simulation/SO101/README.md`, `Simulation/SO101/so101_new_calib.urdf`, `Simulation/SO101/so101_old_calib.urdf`, `Simulation/SO101/assets`
- Local STEP source: `$HOME/Documents/2026 알파프/SO101 Assembly-link11.step`
- Local STEP copy: `simulation/source_step/SO101 Assembly-link11.step`
- STEP status: copied and header/file type checked. No CAD kernel or mesh converter was available, so no STL/DAE conversion was produced.

## Readiness Matrix

| Item | Status | Evidence |
| --- | --- | --- |
| Existing Python FK/IK, sampling, trajectory, scripts, tests remain usable | PASS | `python3 -m pytest tests -q` passed 27 tests. |
| P1 one-link target generation works | PASS | `p1_targets_pilot.csv` generated with configurable target count. Current prepared file has 30 targets. |
| P1 trial schedule generation works | PASS | `p1_trial_schedule.csv` generated with 90 trials for 30 targets x 3 repeats. Adjacent same target count is 0. |
| P1 minimum-jerk trajectory generation works | PASS | `p1_trajectories.csv` generated with smooth return-to-basic segments by default. Current prepared file has 24840 samples. Tests verify q_basic start, target reach, smooth return behavior, monotonic time, minimum-jerk endpoints, and FK columns. |
| ROS2 package builds with colcon | PASS | `colcon build --symlink-install` completed with 1 package finished. |
| URDF/Xacro for P1 one-link model is valid | PASS | Xacro expands and ROS package pytest passed 4 geometry tests. |
| `robot_state_publisher` can publish the model | PASS_WITH_WARNING | Launch starts and `robot_state_publisher` reports `Robot initialized`. Warning: local `joint_state_publisher` is unavailable, so the launch uses the P1 simulation fallback joint-state node. |
| `link1_tip` and `tool0` frames are correctly placed | PASS | URDF test checks `link1_tip_joint` at x = 0.274123 and `tool0_joint` at zero offset. `tf2_echo base_link tool0` showed `[0.000, 0.000, 0.274]` for q2 = pi/2 and `[0.000, 0.274, 0.000]` for q1 = pi/2, q2 = 0. |
| RViz can show robot, tool0, and target markers if GUI is available | PASS_WITH_WARNING | `rviz2` started, OpenGL initialized, and marker publisher loaded 30 target markers. Warning: `joint_state_publisher_gui` is unavailable, so display launch uses the P1 fallback joint-state node. |
| Gazebo launch/spawn is prepared and tested if available | PASS_WITH_WARNING | `ros_gz_sim` was available; `create` reported `Entity creation successful`. Warning: Gazebo printed local EGL/DRI GUI warnings. |
| Verification report exists | PASS | This file. |

## Commands Run

Python/P1:

```bash
cd "$HOME/Documents/2026 알파프/flex_ik_solution"
python3 scripts/check_p1_fk.py
python3 scripts/generate_p1_forward_targets.py \
  --config config/p1_forward_workspace.yaml \
  --output outputs/pre_imu_p1/p1_targets_pilot.csv \
  --metadata-output outputs/pre_imu_p1/p1_targets_pilot_metadata.json
python3 scripts/generate_p1_trial_schedule.py \
  --targets outputs/pre_imu_p1/p1_targets_pilot.csv \
  --repeats 3 \
  --output outputs/pre_imu_p1/p1_trial_schedule.csv \
  --metadata-output outputs/pre_imu_p1/p1_trial_schedule_metadata.json
python3 scripts/generate_p1_trajectories.py \
  --schedule outputs/pre_imu_p1/p1_trial_schedule.csv \
  --output outputs/pre_imu_p1/p1_trajectories.csv
python3 -m pytest tests
```

Python/P1 results:

- FK check: PASS for all three required FK cases.
- Target generation: PASS, configurable target count. Current prepared target count is 30.
- Schedule generation: PASS, 90 trials.
- Trajectory generation: PASS, 24840 samples with smooth return-to-basic segments by default.
- Python tests: PASS, 27 passed.

ROS2 detection:

```bash
echo "$ROS_DISTRO"
which ros2
ros2 pkg prefix gazebo_ros || true
ros2 pkg prefix ros_gz_sim || true
ros2 pkg list | grep -E "gazebo|ros_gz|robot_state_publisher|joint_state_publisher|xacro" || true
```

ROS2 workspace:

```bash
cd "$HOME/Documents/2026 알파프/flex_ik_solution/simulation/ros2_ws"
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
python3 -m pytest src/alphaf_p1_sim/test
```

ROS2 workspace results:

- `rosdep install`: PASS after removing non-required optional joint-state GUI dependencies and invalid `pytest` rosdep key from `package.xml`.
- `colcon build --symlink-install`: PASS.
- ROS package tests: PASS, 4 passed.

Launch checks used short timeouts to avoid leaving long-running simulation processes:

```bash
ros2 launch alphaf_p1_sim state_publisher_p1.launch.py
ros2 run tf2_ros tf2_echo base_link tool0
ros2 launch alphaf_p1_sim display_p1_one_link.launch.py
ros2 launch alphaf_p1_sim gazebo_p1_one_link.launch.py
```

Launch results:

- `state_publisher_p1.launch.py`: PASS_WITH_WARNING. It starts `robot_state_publisher` and uses `p1_joint_state_test.py` because `joint_state_publisher` is not installed.
- `tf2_echo base_link tool0`: PASS. Transform output reached the expected P1 validation positions.
- `display_p1_one_link.launch.py`: PASS_WITH_WARNING. RViz started and target marker node loaded 30 markers; `joint_state_publisher_gui` is missing, so fallback joint states were used.
- `gazebo_p1_one_link.launch.py`: PASS_WITH_WARNING. `ros_gz_sim create` reported entity creation successful; local EGL/DRI GUI warnings appeared.

## Issues Found And Fixed

- Initial P1 target script import failed because `sampling/__init__.py` eagerly imported pandas-backed modules. Fixed by making those package exports lazy.
- Local Python did not have pandas, which the existing 3DoF tests require. Installed pandas into project-local `.python_deps` and added `pytest.ini` pythonpath, without modifying system Python.
- ROS pytest auto-loaded launch plugins that conflict with plain unit collection in this environment. Added pytest config to disable `launch_testing` and `launch_ros` for these unit tests.
- Initial ROS launch failed because shell xacro substitution split a path containing a space/Korean characters. Fixed launch files to use `xacro.process_file(...)`.
- Initial Gazebo spawn reported URDF-to-SDF errors when `base_link` lacked inertia. Fixed by adding a massless technical `p1_model_root` above `base_link`, allowing `base_link` to keep placeholder inertia without triggering KDL root-inertia warnings.
- Initial trajectory preview jumped from target back to q_basic at the next trial boundary. Fixed by adding smooth return-to-basic generation and making the CLI/config default generate `basic -> target -> hold -> basic -> hold` trajectories.
- P1 target generation previously required editing YAML for target count. Fixed by adding `--target-count` and equal occupied-bin allocation metadata.

## Remaining Issues

- CAD mesh conversion is NOT_TESTED because no FreeCAD/OCC/cadquery/mesh conversion tool was available.
- `joint_state_publisher` and `joint_state_publisher_gui` are not installed locally. Launch files use simulation-only fallback joint-state publishing.
- Gazebo runs through `ros_gz_sim`, not classic `gazebo_ros`, because `gazebo_ros` is unavailable.
- Gazebo/RViz GUI startup produced local graphics warnings, but both processes started and Gazebo entity creation succeeded.
- Before uploading to git, keep generated local dependency and build artifact folders ignored: `.python_deps/`, `simulation/ros2_ws/build/`, `simulation/ros2_ws/install/`, and `simulation/ros2_ws/log/`.

## Reproduction Steps

1. Generate P1 targets, schedule, and trajectories with the Python/P1 commands above.
2. Run `python3 -m pytest tests` from the project root.
3. Build the ROS2 workspace with `colcon build --symlink-install`.
4. Source `simulation/ros2_ws/install/setup.bash`.
5. Run `ros2 launch alphaf_p1_sim state_publisher_p1.launch.py`.
6. In another terminal, run `ros2 run tf2_ros tf2_echo base_link tool0`.
7. If GUI is available, run `ros2 launch alphaf_p1_sim display_p1_one_link.launch.py`.
8. If Gazebo is available, run `ros2 launch alphaf_p1_sim gazebo_p1_one_link.launch.py`.
