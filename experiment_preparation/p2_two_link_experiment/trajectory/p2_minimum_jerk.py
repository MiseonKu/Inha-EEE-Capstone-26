"""Minimum-jerk trajectory generation for P2."""

from __future__ import annotations

import math
from pathlib import Path

import hashlib
import json
import numpy as np
import pandas as pd

from kinematics.p2_fk import P2Geometry, fk_p2, load_geometry


PHASES = ("move_to_target", "target_hold", "return_to_basic", "basic_hold")


def minimum_jerk_segment(q_start: np.ndarray, q_goal: np.ndarray, duration_s: float, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return time and q samples for a minimum-jerk segment."""

    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("duration_s and dt_s must be positive")
    t = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    tau = np.clip(t / duration_s, 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    q = q_start[None, :] + s[:, None] * (q_goal - q_start)[None, :]
    return t, q


def generate_trial_trajectory(
    trial: pd.Series,
    geom: P2Geometry,
    *,
    q_basic: np.ndarray | None = None,
    move_duration_s: float = 2.0,
    target_hold_s: float = 1.0,
    return_duration_s: float = 2.0,
    basic_hold_s: float = 0.5,
    sample_dt_s: float = 0.02,
) -> pd.DataFrame:
    """Generate basic -> target -> hold -> basic -> hold rows for one trial."""

    q_basic = np.asarray(q_basic if q_basic is not None else geom.q_basic_rad, dtype=float)
    q_target = np.array([trial["q1_rad"], trial["q2_rad"], trial["q3_rad"]], dtype=float)
    rows = []
    t_offset = 0.0

    def append_phase(phase: str, times: np.ndarray, q_values: np.ndarray) -> None:
        nonlocal rows, t_offset
        for local_t, q in zip(times, q_values):
            p = fk_p2(q, geom)
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "target_id": trial["target_id"],
                    "phase": phase,
                    "t_s": t_offset + float(local_t),
                    "q1_rad": q[0],
                    "q2_rad": q[1],
                    "q3_rad": q[2],
                    "q1_deg": math.degrees(q[0]),
                    "q2_deg": math.degrees(q[1]),
                    "q3_deg": math.degrees(q[2]),
                    "p_tool0_base_x_m": p[0],
                    "p_tool0_base_y_m": p[1],
                    "p_tool0_base_z_m": p[2],
                    "p_tool0_base_x_mm": p[0] * 1000.0,
                    "p_tool0_base_y_mm": p[1] * 1000.0,
                    "p_tool0_base_z_mm": p[2] * 1000.0,
                }
            )
        t_offset = rows[-1]["t_s"] + sample_dt_s

    t, q = minimum_jerk_segment(q_basic, q_target, move_duration_s, sample_dt_s)
    append_phase("move_to_target", t, q)
    hold_t = np.arange(0.0, target_hold_s + 0.5 * sample_dt_s, sample_dt_s)
    append_phase("target_hold", hold_t, np.repeat(q_target[None, :], len(hold_t), axis=0))
    t, q = minimum_jerk_segment(q_target, q_basic, return_duration_s, sample_dt_s)
    append_phase("return_to_basic", t, q)
    hold_t = np.arange(0.0, basic_hold_s + 0.5 * sample_dt_s, sample_dt_s)
    append_phase("basic_hold", hold_t, np.repeat(q_basic[None, :], len(hold_t), axis=0))
    return pd.DataFrame(rows)


def generate_trajectories(
    schedule: pd.DataFrame,
    geometry_path: str | Path,
    *,
    max_trials: int | None = None,
    move_duration_s: float = 2.0,
    target_hold_s: float = 1.0,
    return_duration_s: float = 2.0,
    basic_hold_s: float = 0.5,
    sample_dt_s: float = 0.02,
) -> pd.DataFrame:
    """Generate trajectories for a schedule."""

    geom = load_geometry(geometry_path)
    rows = []
    source = schedule if max_trials is None else schedule.head(max_trials)
    for _, trial in source.iterrows():
        rows.append(
            generate_trial_trajectory(
                trial,
                geom,
                move_duration_s=move_duration_s,
                target_hold_s=target_hold_s,
                return_duration_s=return_duration_s,
                basic_hold_s=basic_hold_s,
                sample_dt_s=sample_dt_s,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_trajectories(trajectories: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output, index=False)


def trajectory_metadata(
    trajectories: pd.DataFrame,
    geom: P2Geometry,
    *,
    target_generation_version: str | None = None,
) -> dict:
    """Build trajectory metadata for downstream validation."""

    return {
        "stage": geom.stage,
        "coordinate_frame": geom.base_frame,
        "target_count": int(trajectories["target_id"].nunique()) if len(trajectories) else 0,
        "trial_count": int(trajectories["trial_id"].nunique()) if len(trajectories) else 0,
        "row_count": int(len(trajectories)),
        "geometry": {
            "L1_effective_m": geom.L1_effective_m,
            "L2_effective_m": geom.L2_effective_m,
            "link1_flexible_bar_length_m": geom.link1_flexible_bar_length_m,
            "link2_flexible_bar_length_m": geom.link2_flexible_bar_length_m,
            "material_preset": geom.material_preset,
        },
        "target_generation_version": target_generation_version,
        "trajectory_generation_version": "p2_minimum_jerk_v1",
        "label_policy": "trajectory contains commanded/reference motion only; Vive Tracker settled position is label",
        "csv_sha256": hashlib.sha256(trajectories.to_csv(index=False).encode("utf-8")).hexdigest(),
    }


def write_trajectory_metadata(metadata: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
