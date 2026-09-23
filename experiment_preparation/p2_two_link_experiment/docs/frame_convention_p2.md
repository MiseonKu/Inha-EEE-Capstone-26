# Frame Convention P2

`base_link` AlphaF convention:

- +X forward, away from desk
- +Y lateral
- +Z upward

Joint convention:

- q1: base yaw around +Z
- q2: shoulder pitch
- q3: elbow pitch

Rigid FK:

```text
rho = L1*cos(q2) + L2*cos(q2 + q3)
z   = L1*sin(q2) + L2*sin(q2 + q3)
x   = rho*cos(q1)
y   = rho*sin(q1)
```

Validation cases are implemented in `scripts/check_p2_fk.py`.

Basic pose:

```text
q_basic = [0, pi/2, -pi/2]
tool0 = [0.29945, 0, 0.2996]
```

Before real hardware execution, motor sign, encoder zero, and radian scale must be verified.
