"""Future Vive-to-base calibration scaffold for P2.

Required input columns:
- target_id
- p_base_x_mm, p_base_y_mm, p_base_z_mm
- p_vive_x_mm, p_vive_y_mm, p_vive_z_mm

This is offline parsing only and never connects to Vive hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = ["target_id", "p_base_x_mm", "p_base_y_mm", "p_base_z_mm", "p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate simple Vive-to-base translation from calibration pairs.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pre_imu_p2/p2_vive_to_base_calibration.json"))
    args = parser.parse_args()
    df = pd.read_csv(args.pairs)
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    delta = df[["p_base_x_mm", "p_base_y_mm", "p_base_z_mm"]].to_numpy() - df[["p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"]].to_numpy()
    payload = {
        "method": "translation_only_scaffold",
        "status": "PASS_WITH_WARNING",
        "warning": "Rotation calibration is not implemented yet.",
        "translation_vive_to_base_mm": delta.mean(axis=0).tolist(),
        "pair_count": int(len(df)),
        "required_columns": REQUIRED,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
