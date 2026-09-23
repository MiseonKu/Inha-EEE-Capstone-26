# AlphaF P1 ROS2 Simulation Workspace

This workspace contains the simulation-only `alphaf_p1_sim` package for the
P1 one-link rigid reference setup. It does not contain firmware upload,
serial, calibration, torque, or motor command tooling.

```bash
cd simulation/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch alphaf_p1_sim state_publisher_p1.launch.py
```
