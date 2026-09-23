# P2 Experiment Setup

P2 uses the SO-101 base with two installed links. The experiment target is the rigid FK reference position and the real label is the Vive Tracker settled position measured at the link2 tip/tool0.

Sequence:

1. Start from `q_basic = [0, pi/2, -pi/2]`.
2. Move with a minimum-jerk trajectory to a generated target q.
3. Hold at target for settling.
4. Return smoothly to `q_basic`.
5. Hold at basic pose before the next trial.

No IMU is used in this phase. Vive Tracker is not control feedback. This package prepares target, schedule, trajectory, plots, and metadata only.
