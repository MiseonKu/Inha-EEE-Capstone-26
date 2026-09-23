"""Generate P2 basic-target-hold-basic-hold minimum-jerk trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p2_fk import load_geometry  # noqa: E402
from trajectory.p2_minimum_jerk import generate_trajectories, trajectory_metadata, write_trajectories, write_trajectory_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P2 minimum-jerk trajectories.")
    parser.add_argument("--schedule", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trial_schedule.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "config" / "p2_geometry.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trajectories.csv")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trajectories_metadata.json")
    parser.add_argument("--move-duration-s", type=float, default=2.0)
    parser.add_argument("--target-hold-s", type=float, default=1.0)
    parser.add_argument("--return-duration-s", type=float, default=2.0)
    parser.add_argument("--basic-hold-s", type=float, default=0.5)
    parser.add_argument("--sample-dt-s", type=float, default=0.02)
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()
    schedule = pd.read_csv(args.schedule)
    trajectories = generate_trajectories(
        schedule,
        args.geometry,
        max_trials=args.max_trials,
        move_duration_s=args.move_duration_s,
        target_hold_s=args.target_hold_s,
        return_duration_s=args.return_duration_s,
        basic_hold_s=args.basic_hold_s,
        sample_dt_s=args.sample_dt_s,
    )
    write_trajectories(trajectories, args.output)
    geom = load_geometry(args.geometry)
    version = None
    if "target_generation_version" in schedule.columns:
        version = str(schedule["target_generation_version"].iloc[0])
    if version is None:
        version = "p2_forward_targets_v1"
    write_trajectory_metadata(trajectory_metadata(trajectories, geom, target_generation_version=version), args.metadata_output)
    print(f"wrote {args.output}")
    print(f"wrote {args.metadata_output}")
    print(f"trajectory_rows={len(trajectories)} trials={trajectories['trial_id'].nunique() if len(trajectories) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
