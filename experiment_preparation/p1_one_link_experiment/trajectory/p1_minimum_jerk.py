"""P1 two-joint minimum-jerk trajectory generation."""

from __future__ import annotations

import csv
from math import degrees, pi
from pathlib import Path
from typing import Any

import numpy as np

from kinematics.p1_fk import Q_BASIC, fk_p1


TRAJECTORY_COLUMNS = [
    "trial_id",
    "target_id",
    "t_s",
    "q1_rad",
    "q2_rad",
    "q1_deg",
    "q2_deg",
    "p_tool0_base_x_m",
    "p_tool0_base_y_m",
    "p_tool0_base_z_m",
]


def minimum_jerk_s(tau: np.ndarray | float) -> np.ndarray | float:
    """Minimum-jerk scalar progress."""

    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def read_schedule(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["trial_id", "target_id"]:
            row[key] = int(row[key])
        for key in ["q1_rad", "q2_rad"]:
            row[key] = float(row[key])
    return rows


def generate_p1_trajectories(
    schedule: list[dict[str, Any]],
    q_basic: np.ndarray = Q_BASIC,
    move_duration_s: float = 2.0,
    sample_dt_s: float = 0.02,
    include_return: bool = False,
    return_duration_s: float | None = None,
    target_hold_s: float = 0.0,
    basic_hold_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate all P1 trial trajectories."""

    if move_duration_s <= 0.0:
        raise ValueError("move_duration_s must be positive")
    if sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be positive")
    if return_duration_s is None:
        return_duration_s = move_duration_s
    if return_duration_s <= 0.0:
        raise ValueError("return_duration_s must be positive")
    if target_hold_s < 0.0 or basic_hold_s < 0.0:
        raise ValueError("hold durations must be non-negative")

    rows: list[dict[str, Any]] = []
    q0 = np.asarray(q_basic, dtype=float).reshape(2)
    for trial in schedule:
        q_goal = np.array([float(trial["q1_rad"]), float(trial["q2_rad"])], dtype=float)
        trial_rows: list[dict[str, Any]] = []
        _append_move(
            trial_rows,
            trial=trial,
            q_start=q0,
            q_goal=q_goal,
            t_start=0.0,
            duration_s=move_duration_s,
            sample_dt_s=sample_dt_s,
            skip_first=False,
        )
        t_cursor = float(trial_rows[-1]["t_s"])
        if target_hold_s > 0.0:
            t_cursor = _append_hold(
                trial_rows,
                trial=trial,
                q_hold=q_goal,
                t_start=t_cursor,
                hold_s=target_hold_s,
                sample_dt_s=sample_dt_s,
            )
        if include_return:
            _append_move(
                trial_rows,
                trial=trial,
                q_start=q_goal,
                q_goal=q0,
                t_start=t_cursor,
                duration_s=return_duration_s,
                sample_dt_s=sample_dt_s,
                skip_first=True,
            )
            t_cursor = float(trial_rows[-1]["t_s"])
            if basic_hold_s > 0.0:
                _append_hold(
                    trial_rows,
                    trial=trial,
                    q_hold=q0,
                    t_start=t_cursor,
                    hold_s=basic_hold_s,
                    sample_dt_s=sample_dt_s,
                )
        rows.extend(trial_rows)
    return rows


def _row(trial: dict[str, Any], t_s: float, q_row: np.ndarray) -> dict[str, Any]:
    p = fk_p1(q_row)
    return {
        "trial_id": int(trial["trial_id"]),
        "target_id": int(trial["target_id"]),
        "t_s": float(t_s),
        "q1_rad": float(q_row[0]),
        "q2_rad": float(q_row[1]),
        "q1_deg": degrees(float(q_row[0])),
        "q2_deg": degrees(float(q_row[1])),
        "p_tool0_base_x_m": float(p[0]),
        "p_tool0_base_y_m": float(p[1]),
        "p_tool0_base_z_m": float(p[2]),
    }


def _append_move(
    rows: list[dict[str, Any]],
    trial: dict[str, Any],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    t_start: float,
    duration_s: float,
    sample_dt_s: float,
    skip_first: bool,
) -> None:
    n_steps = int(round(duration_s / sample_dt_s)) + 1
    local_time = np.linspace(0.0, duration_s, n_steps, dtype=float)
    tau = local_time / duration_s
    s = minimum_jerk_s(tau)
    q = q_start[None, :] + s[:, None] * (q_goal - q_start)[None, :]
    start_index = 1 if skip_first else 0
    for t_local, q_row in zip(local_time[start_index:], q[start_index:]):
        rows.append(_row(trial, t_start + float(t_local), q_row))


def _append_hold(
    rows: list[dict[str, Any]],
    trial: dict[str, Any],
    q_hold: np.ndarray,
    t_start: float,
    hold_s: float,
    sample_dt_s: float,
) -> float:
    n_steps = int(round(hold_s / sample_dt_s))
    t_cursor = t_start
    for step in range(1, n_steps + 1):
        t_cursor = t_start + step * sample_dt_s
        rows.append(_row(trial, t_cursor, q_hold))
    return t_cursor


def write_p1_trajectories(rows: list[dict[str, Any]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in TRAJECTORY_COLUMNS})
