from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p2_fk import fk_p2, load_geometry
from sampling.p2_forward_targets import generate_targets
from sampling.p2_trial_schedule import generate_schedule
from trajectory.p2_minimum_jerk import PHASES, generate_trajectories


def test_trajectory_generation_contract() -> None:
    geom = load_geometry(ROOT / "config" / "p2_geometry.yaml")
    targets, _ = generate_targets(
        ROOT / "config" / "p2_geometry.yaml",
        ROOT / "config" / "p2_forward_workspace.yaml",
        target_count=5,
        candidate_count=500,
        seed=21,
    )
    schedule, _ = generate_schedule(targets, repeats=1, seed=22)
    traj = generate_trajectories(schedule, ROOT / "config" / "p2_geometry.yaml", max_trials=2, sample_dt_s=0.1)
    first = traj[traj["trial_id"] == traj["trial_id"].iloc[0]]
    assert first["phase"].iloc[0] == "move_to_target"
    assert set(PHASES).issubset(set(first["phase"]))
    assert np.all(np.diff(first["t_s"]) > -1e-12)
    q_start = first.iloc[0][["q1_rad", "q2_rad", "q3_rad"]].to_numpy(dtype=float)
    q_end = first.iloc[-1][["q1_rad", "q2_rad", "q3_rad"]].to_numpy(dtype=float)
    assert np.allclose(q_start, np.array(geom.q_basic_rad), atol=1e-12)
    assert np.allclose(q_end, np.array(geom.q_basic_rad), atol=1e-12)
    hold = first[first["phase"] == "target_hold"].iloc[0]
    target = schedule[schedule["trial_id"] == first["trial_id"].iloc[0]].iloc[0]
    assert np.allclose(hold[["q1_rad", "q2_rad", "q3_rad"]].to_numpy(dtype=float), target[["q1_rad", "q2_rad", "q3_rad"]].to_numpy(dtype=float))
    sample = first.iloc[len(first) // 2]
    p = fk_p2(sample[["q1_rad", "q2_rad", "q3_rad"]].to_numpy(dtype=float), geom)
    assert np.allclose(p, sample[["p_tool0_base_x_m", "p_tool0_base_y_m", "p_tool0_base_z_m"]].to_numpy(dtype=float), atol=1e-12)
