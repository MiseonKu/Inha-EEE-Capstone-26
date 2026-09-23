# P1 Frame Convention

AlphaF base convention:

- `base_link`: +X forward, +Y lateral, +Z up.
- `joint_1_yaw`: base yaw about +Z. q1 = 0 points forward.
- `yaw_link`: frame after q1.
- `joint_2_pitch`: shoulder pitch. The URDF axis is `0 -1 0` so positive q2 rotates link local +X toward base +Z.
- `pitch_link`: frame after q2.
- `link_1_custom`: P1 one-link body frame, colocated with the q2 origin.
- `link1_tip`: fixed at x = 0.274123 m in `link_1_custom`.
- `tool0`: fixed and colocated with `link1_tip`.

The URDF also contains a massless technical root link named `p1_model_root`
fixed to `base_link`. It is not an experiment frame.

Rigid FK convention:

```text
rho = L1 * cos(q2)
z   = L1 * sin(q2)
x   = rho * cos(q1)
y   = rho * sin(q1)
```

Validation cases:

- q1 = 0, q2 = 0: `link1_tip = [0.274123, 0, 0]`.
- q1 = 0, q2 = pi/2: `link1_tip = [0, 0, 0.274123]`.
- q1 = pi/2, q2 = 0: `link1_tip = [0, 0.274123, 0]`.

The motor sign convention still must be checked before any later real hardware
execution. This package is simulation-side preparation only.
