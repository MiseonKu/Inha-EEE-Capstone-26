"""Generate P2 forward-hemisphere pilot targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p2_forward_targets import generate_targets, write_targets  # noqa: E402


def pair(values: list[str]) -> list[float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range requires LOW HIGH")
    return [float(values[0]), float(values[1])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate P2 forward trimmed hemisphere targets.")
    parser.add_argument("--geometry", type=Path, default=ROOT / "config" / "p2_geometry.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "p2_forward_workspace.yaml")
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--q1-range-deg", nargs=2, default=None)
    parser.add_argument("--q2-range-deg", nargs=2, default=None)
    parser.add_argument("--q3-range-deg", nargs=2, default=None)
    parser.add_argument("--azimuth-range-deg", nargs=2, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_targets_pilot.csv")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_targets_pilot_metadata.json")
    parser.add_argument("--preview-output", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_workspace_preview.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    q_ranges = None
    if args.q1_range_deg or args.q2_range_deg or args.q3_range_deg:
        from kinematics.p2_fk import load_yaml

        cfg = load_yaml(args.config)
        q_ranges = {key: [float(v[0]), float(v[1])] for key, v in cfg["q_ranges_deg"].items()}
        for key, values in (("q1", args.q1_range_deg), ("q2", args.q2_range_deg), ("q3", args.q3_range_deg)):
            if values:
                q_ranges[key] = [float(values[0]), float(values[1])]
    azimuth = [float(v) for v in args.azimuth_range_deg] if args.azimuth_range_deg else None
    targets, metadata = generate_targets(
        args.geometry,
        args.config,
        target_count=args.target_count,
        candidate_count=args.candidate_count,
        seed=args.seed,
        q_ranges_deg=q_ranges,
        azimuth_range_deg=azimuth,
    )
    write_targets(targets, metadata, args.output, args.metadata_output)
    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
    targets[["target_id", "p_target_base_x_m", "p_target_base_y_m", "p_target_base_z_m", "azimuth_deg", "radius_m", "height_m"]].to_csv(args.preview_output, index=False)
    print(f"wrote {args.output}")
    print(f"wrote {args.preview_output}")
    print(f"targets={len(targets)} occupied_bins={metadata['occupied_bin_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
