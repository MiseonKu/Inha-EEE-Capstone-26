"""Future P2 settled dataset builder scaffold.

This script validates input schemas and can merge target/schedule metadata.
It does not assume live hardware logs are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a P2 trial-level settled dataset from offline logs.")
    parser.add_argument("--targets", type=Path, default=Path("outputs/pre_imu_p2/p2_targets_pilot.csv"))
    parser.add_argument("--schedule", type=Path, default=Path("outputs/pre_imu_p2/p2_trial_schedule.csv"))
    parser.add_argument("--vive-log", type=Path, default=None)
    parser.add_argument("--encoder-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/pre_imu_p2/p2_settled_dataset.csv"))
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    schedule = pd.read_csv(args.schedule)
    merged = schedule.merge(targets[["target_id", "stage", "material_preset", "target_generation_version"]], on="target_id", how="left")
    merged["label_available"] = False
    merged["quality_flags"] = "NO_REAL_VIVE_LOG"
    if args.vive_log and args.vive_log.exists():
        merged["quality_flags"] = "VIVE_LOG_PRESENT_NOT_YET_ALIGNED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    meta = {
        "status": "PASS_WITH_WARNING",
        "warning": "Real Vive/encoder alignment is scaffolded but not implemented without actual log contract.",
        "rows": int(len(merged)),
        "vive_log": str(args.vive_log) if args.vive_log else None,
        "encoder_log": str(args.encoder_log) if args.encoder_log else None,
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
