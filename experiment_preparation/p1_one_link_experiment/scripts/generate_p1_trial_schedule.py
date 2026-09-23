#!/usr/bin/env python3
"""Generate a balanced P1 trial schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p1_trial_schedule import (
    generate_p1_schedule,
    read_targets,
    write_p1_schedule,
    write_p1_schedule_metadata,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate balanced P1 trial schedule.")
    parser.add_argument("--targets", default="outputs/pre_imu_p1/p1_targets_pilot.csv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--output", default="outputs/pre_imu_p1/p1_trial_schedule.csv")
    parser.add_argument("--metadata-output", default="outputs/pre_imu_p1/p1_trial_schedule_metadata.json")
    args = parser.parse_args()

    targets = read_targets(_resolve(args.targets))
    schedule = generate_p1_schedule(targets, repeats=args.repeats, seed=args.seed)
    write_p1_schedule(schedule, _resolve(args.output))
    write_p1_schedule_metadata(schedule, targets, args.repeats, args.seed, _resolve(args.metadata_output))
    print(f"wrote {_resolve(args.output)}")
    print(f"wrote {_resolve(args.metadata_output)}")
    print(f"trials={len(schedule)} targets={len(targets)} repeats={args.repeats} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
