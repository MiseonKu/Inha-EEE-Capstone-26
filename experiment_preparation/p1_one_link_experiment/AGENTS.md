# AGENTS

이 폴더는 AlphaF P1 one-link 실험의 simulation/data collection preparation 전용 workspace입니다.

## Scope

- ROS2/RViz2/Gazebo simulation assets
- P1 rigid FK validation
- target/schedule/trajectory CSV generation
- Vive Tracker coordinate comparison scaffold
- offline dataset preparation utilities

## Safety Rules

- Do not upload firmware.
- Do not open a serial monitor.
- Do not send torque, INIT, calibration, MOVE, or motor commands.
- Do not use Vive Tracker as closed-loop control feedback. Tracker data is a ground-truth label/evaluation signal only.
- Generated real experiment logs should stay local unless the user explicitly asks to archive them.

## Main Commands

```bash
python3 scripts/check_p1_fk.py
bash scripts/p1_prepare_only.sh --target-count 120 --repeats 3
python3 -m pytest tests -q
```

ROS2:

```bash
cd simulation/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
python3 -m pytest src/alphaf_p1_sim/test -q
```

## Notes

The authoritative P1 FK length is `L1_effective_m = 0.274123`. Do not infer FK length or joint sign from the STEP file.
