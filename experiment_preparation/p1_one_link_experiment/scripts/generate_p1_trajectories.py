#!/usr/bin/env python3
"""Generate P1 minimum-jerk trajectories from q_basic to scheduled targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p1_fk import Q_BASIC
from trajectory.p1_minimum_jerk import generate_p1_trajectories, read_schedule, write_p1_trajectories


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _trajectory_defaults(config_path: Path | None) -> dict[str, object]:
    if config_path is None or not config_path.exists():
        return {
            "q_basic": Q_BASIC.copy(),
            "move_duration_s": 2.0,
            "sample_dt_s": 0.02,
            "include_return": True,
            "return_duration_s": 2.0,
            "target_hold_s": 1.0,
            "basic_hold_s": 0.5,
        }
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    trajectory = data.get("trajectory", {})
    return {
        "q_basic": np.asarray(trajectory.get("q_basic_rad", Q_BASIC.tolist()), dtype=float),
        "move_duration_s": float(trajectory.get("move_duration_s", 2.0)),
        "sample_dt_s": float(trajectory.get("sample_dt_s", 0.02)),
        "include_return": bool(trajectory.get("include_return", True)),
        "return_duration_s": float(trajectory.get("return_duration_s", trajectory.get("move_duration_s", 2.0))),
        "target_hold_s": float(trajectory.get("target_hold_s", 1.0)),
        "basic_hold_s": float(trajectory.get("basic_hold_s", 0.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P1 minimum-jerk trajectories.")
    parser.add_argument("--schedule", default="outputs/pre_imu_p1/p1_trial_schedule.csv")
    parser.add_argument("--output", default="outputs/pre_imu_p1/p1_trajectories.csv")
    parser.add_argument("--config", default="config/p1_forward_workspace.yaml")
    parser.add_argument("--move-duration-s", type=float, default=None)
    parser.add_argument("--sample-dt-s", type=float, default=None)
    parser.add_argument("--include-return", dest="include_return", action="store_true", default=None)
    parser.add_argument("--no-include-return", dest="include_return", action="store_false")
    parser.add_argument("--return-duration-s", type=float, default=None)
    parser.add_argument("--target-hold-s", type=float, default=None)
    parser.add_argument("--basic-hold-s", type=float, default=None)
    args = parser.parse_args()

    defaults = _trajectory_defaults(_resolve(args.config))
    q_basic = defaults["q_basic"]
    duration = float(defaults["move_duration_s"])
    dt = float(defaults["sample_dt_s"])
    include_return = bool(defaults["include_return"])
    return_duration = float(defaults["return_duration_s"])
    target_hold = float(defaults["target_hold_s"])
    basic_hold = float(defaults["basic_hold_s"])
    if args.move_duration_s is not None:
        duration = args.move_duration_s
    if args.sample_dt_s is not None:
        dt = args.sample_dt_s
    if args.include_return is not None:
        include_return = args.include_return
    if args.return_duration_s is not None:
        return_duration = args.return_duration_s
    if args.target_hold_s is not None:
        target_hold = args.target_hold_s
    if args.basic_hold_s is not None:
        basic_hold = args.basic_hold_s

    schedule = read_schedule(_resolve(args.schedule))
    trajectories = generate_p1_trajectories(
        schedule,
        q_basic=q_basic,
        move_duration_s=duration,
        sample_dt_s=dt,
        include_return=include_return,
        return_duration_s=return_duration,
        target_hold_s=target_hold,
        basic_hold_s=basic_hold,
    )
    write_p1_trajectories(trajectories, _resolve(args.output))
    print(f"wrote {_resolve(args.output)}")
    print(
        f"trials={len(schedule)} samples={len(trajectories)} duration_s={duration} "
        f"dt_s={dt} include_return={include_return} return_duration_s={return_duration} "
        f"target_hold_s={target_hold} basic_hold_s={basic_hold}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
