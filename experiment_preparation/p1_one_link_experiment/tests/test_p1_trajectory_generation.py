from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p1_fk import Q_BASIC, fk_p1
from sampling.p1_forward_targets import generate_p1_targets, load_p1_target_config
from sampling.p1_trial_schedule import generate_p1_schedule
from trajectory.p1_minimum_jerk import generate_p1_trajectories, minimum_jerk_s


def _small_schedule():
    config = load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml")
    targets = generate_p1_targets(config)[:4]
    return generate_p1_schedule(targets, repeats=1, seed=202)


def test_p1_trajectory_starts_and_ends_at_requested_q():
    schedule = _small_schedule()
    rows = generate_p1_trajectories(schedule, move_duration_s=2.0, sample_dt_s=0.02)
    by_trial = defaultdict(list)
    for row in rows:
        by_trial[row["trial_id"]].append(row)

    for trial in schedule:
        trajectory = by_trial[trial["trial_id"]]
        assert np.allclose([trajectory[0]["q1_rad"], trajectory[0]["q2_rad"]], Q_BASIC, atol=1e-12)
        assert np.allclose(
            [trajectory[-1]["q1_rad"], trajectory[-1]["q2_rad"]],
            [trial["q1_rad"], trial["q2_rad"]],
            atol=1e-12,
        )


def test_p1_trajectory_time_is_monotonic_per_trial():
    rows = generate_p1_trajectories(_small_schedule(), move_duration_s=2.0, sample_dt_s=0.02)
    by_trial = defaultdict(list)
    for row in rows:
        by_trial[row["trial_id"]].append(float(row["t_s"]))
    for times in by_trial.values():
        assert all(t1 < t2 for t1, t2 in zip(times, times[1:]))


def test_p1_minimum_jerk_progress_endpoints():
    assert minimum_jerk_s(0.0) == 0.0
    assert minimum_jerk_s(1.0) == 1.0


def test_p1_trajectory_fk_columns_match_q():
    rows = generate_p1_trajectories(_small_schedule(), move_duration_s=2.0, sample_dt_s=0.5)
    for row in rows:
        p = fk_p1([row["q1_rad"], row["q2_rad"]])
        assert np.allclose(
            [row["p_tool0_base_x_m"], row["p_tool0_base_y_m"], row["p_tool0_base_z_m"]],
            p,
            atol=1e-12,
        )


def test_p1_return_trajectory_comes_back_to_basic_without_jump():
    schedule = _small_schedule()
    rows = generate_p1_trajectories(
        schedule,
        move_duration_s=1.0,
        sample_dt_s=0.1,
        include_return=True,
        return_duration_s=1.0,
        target_hold_s=0.2,
        basic_hold_s=0.2,
    )
    by_trial = defaultdict(list)
    for row in rows:
        by_trial[row["trial_id"]].append(row)

    for trial in schedule:
        trajectory = by_trial[trial["trial_id"]]
        assert np.allclose([trajectory[0]["q1_rad"], trajectory[0]["q2_rad"]], Q_BASIC, atol=1e-12)
        assert np.allclose([trajectory[-1]["q1_rad"], trajectory[-1]["q2_rad"]], Q_BASIC, atol=1e-12)
        assert any(
            np.allclose([row["q1_rad"], row["q2_rad"]], [trial["q1_rad"], trial["q2_rad"]], atol=1e-12)
            for row in trajectory
        )
