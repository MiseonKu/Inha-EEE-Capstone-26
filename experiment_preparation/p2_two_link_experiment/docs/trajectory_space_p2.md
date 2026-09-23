# Trajectory Space P2

The trajectory generator uses joint-space minimum jerk:

```text
s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5
q(t) = q_start + s(t) * (q_goal - q_start)
```

Default phases:

- `move_to_target`: basic to target, 2.0 s
- `target_hold`: hold target, 1.0 s
- `return_to_basic`: target to basic, 2.0 s
- `basic_hold`: hold basic, 0.5 s

The return is smooth and no instant target-to-basic jump is inserted.
