#!/usr/bin/env python3
"""Generate P1 pilot forward-hemisphere target CSV and metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p1_forward_targets import (
    generate_p1_targets,
    load_p1_target_config,
    override_p1_target_config,
    write_p1_target_metadata,
    write_p1_targets,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate P1 forward pilot targets.")
    parser.add_argument("--config", default="config/p1_forward_workspace.yaml")
    parser.add_argument("--output", default="outputs/pre_imu_p1/p1_targets_pilot.csv")
    parser.add_argument("--metadata-output", default="outputs/pre_imu_p1/p1_targets_pilot_metadata.json")
    parser.add_argument("--target-count", type=int, default=None, help="Override number of target points to generate.")
    parser.add_argument("--candidate-count", type=int, default=None, help="Override candidate count before bin-balanced selection.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--q1-range-deg", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--q2-range-deg", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--azimuth-range-deg", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--yaw-bins", type=int, default=None)
    parser.add_argument("--pitch-bins", type=int, default=None)
    parser.add_argument("--height-bins", type=int, default=None)
    args = parser.parse_args()

    config = override_p1_target_config(
        load_p1_target_config(_resolve(args.config)),
        target_count=args.target_count,
        candidate_count=args.candidate_count,
        seed=args.seed,
        q1_range_deg=args.q1_range_deg,
        q2_range_deg=args.q2_range_deg,
        azimuth_range_deg=args.azimuth_range_deg,
        yaw_bins=args.yaw_bins,
        pitch_bins=args.pitch_bins,
        height_bins=args.height_bins,
    )
    targets = generate_p1_targets(config)
    write_p1_targets(targets, _resolve(args.output))
    write_p1_target_metadata(config, targets, _resolve(args.metadata_output))
    print(f"wrote {_resolve(args.output)}")
    print(f"wrote {_resolve(args.metadata_output)}")
    print(f"targets={len(targets)} seed={config.seed} workspace={config.workspace_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
