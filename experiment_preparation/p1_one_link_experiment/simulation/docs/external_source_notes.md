# External Source Notes

SO-ARM100/SO-101 reference:

- Source URL: <https://github.com/TheRobotStudio/SO-ARM100>
- Inspected commit: `fda892cba81032c46c40976a48c9ceadbf40a9ca`
- License: Apache-2.0
- Repository default branch: `main`

Inspected upstream files and folders:

- `Simulation/README.md`
- `Simulation/SO101/README.md`
- `Simulation/SO101/so101_new_calib.urdf`
- `Simulation/SO101/so101_old_calib.urdf`
- `Simulation/SO101/assets/*.stl`
- `Simulation/SO101/scene.xml`
- `Simulation/SO101/joints_properties.xml`
- `LICENSE`

Observed upstream conventions:

- Upstream SO101 simulation includes URDF, MJCF, and STL assets.
- Upstream SO101 URDF is generated using `onshape-to-robot`.
- Upstream joint names include `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, and `gripper`.
- Upstream base link name is `base_link`.
- Upstream gripper/tool area includes `gripper_frame_link`.

Copied upstream files:

- None.

Local modifications:

- The local P1 model is a clean URDF/Xacro generated from the explicit AlphaF P1 geometry.
- Only two active joints are kept: `joint_1_yaw` and `joint_2_pitch`.
- Elbow, wrist, and gripper moving joints are not present.
- Simple box visual/collision geometry is used because no CAD conversion tool was available.

Local STEP file:

- Source: `$HOME/Documents/2026 알파프/SO101 Assembly-link11.step`
- Copied to: `simulation/source_step/SO101 Assembly-link11.step`
- Purpose: visual geometry reference only.
- It is not used for FK length, joint axes, pitch sign, or base convention.

Algorithmic source:

- The existing local `flex_ik_solution` modules were reused for project structure and workflow style.
- The requested `03_flexible_link_fkik_workspace_point_algorithm` reference folder was not found in the local `Documents/2026 알파프` search during this pass.
- P1 target selection adapts the same ideas already present in this workspace: joint-space sampling, forward workspace filtering, bin labels, balanced scheduling, and minimum-jerk interpolation.
