# Geometry Authority P2

Authoritative kinematics:

- `L1_effective_m = 0.2996`
- `L2_effective_m = 0.29945`
- `link1_flexible_bar_length_m = 0.233`
- `link2_flexible_bar_length_m = 0.215`
- `link1_nonflex_offset_mm = 66.6`
- `link2_nonflex_offset_mm = 84.45`

`L1_effective` is q2 shoulder pitch axis to q3 elbow pitch axis. `L2_effective` is q3 elbow pitch axis to link2_tip/tool0 unless a later tracker/tool offset is supplied.

The STEP file `source_step/2_LINK_SO101_ASSEMBLY.step` is a visual reference. It is not the source for FK length, joint axes, yaw sign, or pitch sign. Optional CAD measurements are recorded only as observations.
