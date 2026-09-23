"""Generate a repeat-balanced P2 trial schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p2_trial_schedule import generate_schedule, write_schedule  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P2 trial schedule.")
    parser.add_argument("--targets", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_targets_pilot.csv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trial_schedule.csv")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trial_schedule_metadata.json")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    schedule, metadata = generate_schedule(targets, repeats=args.repeats, seed=args.seed)
    write_schedule(schedule, metadata, args.output, args.metadata_output)
    print(f"wrote {args.output}")
    print(f"trials={len(schedule)} targets={schedule['target_id'].nunique()} repeats={args.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
