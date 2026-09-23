# P1 Experiment Setup

Stage: `P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL`.

Physical context:

- Only link 1 is installed.
- Links and joints downstream of link 1 are removed.
- A Vive Tracker is mounted at the end of link 1.
- IMU is unavailable in this phase.
- ROS2/Gazebo provides the rigid reference coordinate.
- The Vive Tracker measured settled position becomes the real label.

Basic pose:

- `q_basic = [q1, q2] = [0, pi/2]`.
- q1 = 0 means forward.
- q2 = pi/2 means link 1 vertical.

Generated preparation files:

- `outputs/pre_imu_p1/p1_targets_pilot.csv`: 30 rigid reference targets.
- `outputs/pre_imu_p1/p1_trial_schedule.csv`: balanced 3-repeat trial order.
- `outputs/pre_imu_p1/p1_trajectories.csv`: minimum-jerk joint trajectories from q_basic to target q.

The target point is not the label. During the later real experiment, each target
is the commanded rigid reference coordinate and the Vive Tracker settled
measurement is the ground-truth label. No Vive or IMU fields are required in the
simulation-side target files.

This workspace does not upload firmware, open serial monitors, calibrate motors,
send torque, send INIT, or send MOVE commands.
