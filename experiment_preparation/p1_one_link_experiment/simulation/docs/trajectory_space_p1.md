# P1 Trajectory Space

Trajectory start and return:

```text
q_basic = [0, pi/2]
```

Target q:

- `q1_rad` and `q2_rad` come from `p1_targets_pilot.csv` through the balanced schedule.
- FK uses `L1_effective = 0.274123 m`.

Minimum-jerk formula:

```text
s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5
q(t) = q_basic + s(t) * (q_target - q_basic)
tau = t / T
```

Defaults:

- `move_duration_s = 2.0`.
- `sample_dt_s = 0.02`.
- Real-experiment trajectory export now includes a smooth return segment after each target:
  `q_basic -> q_target -> target hold -> q_basic -> basic hold`.
- This avoids an instantaneous target-to-basic reset between trials when the sequence is replayed with a Vive Tracker mounted.

Generated columns:

- `trial_id`, `target_id`, `t_s`.
- `q1_rad`, `q2_rad`, `q1_deg`, `q2_deg`.
- `p_tool0_base_x_m`, `p_tool0_base_y_m`, `p_tool0_base_z_m`.

The trajectory CSV is a simulation-side command reference for later experiment
execution. Hardware motor signs, offsets, and safety limits still need real
hardware verification before the user runs the physical experiment.
