#!/usr/bin/env python3
"""Build settled P1 ML dataset rows from live Vive comparison logs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT_COLUMNS = [
    "trial_id",
    "target_id",
    "settle_window_s",
    "sample_count",
    "q1_encoder_rad_mean",
    "q2_encoder_rad_mean",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "p_vive_settled_base_x_m",
    "p_vive_settled_base_y_m",
    "p_vive_settled_base_z_m",
    "error_x_m",
    "error_y_m",
    "error_z_m",
    "error_norm_m",
]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(rows: list[dict[str, str]], key: str) -> float:
    values = np.array([_float_or_nan(row.get(key, "")) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def build_dataset(log_rows: list[dict[str, str]], settle_window_s: float) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in log_rows:
        groups[(int(row["trial_id"]), int(row["target_id"]))].append(row)

    out: list[dict[str, object]] = []
    for (trial_id, target_id), rows in sorted(groups.items()):
        rows.sort(key=lambda row: _float_or_nan(row["stamp_s"]))
        end_time = _float_or_nan(rows[-1]["stamp_s"])
        settled = [row for row in rows if end_time - _float_or_nan(row["stamp_s"]) <= settle_window_s]
        if not settled:
            settled = [rows[-1]]
        target = np.array(
            [
                _float_or_nan(settled[-1]["p_target_base_x_m"]),
                _float_or_nan(settled[-1]["p_target_base_y_m"]),
                _float_or_nan(settled[-1]["p_target_base_z_m"]),
            ],
            dtype=float,
        )
        vive = np.array(
            [
                _mean(settled, "p_vive_base_x_m"),
                _mean(settled, "p_vive_base_y_m"),
                _mean(settled, "p_vive_base_z_m"),
            ],
            dtype=float,
        )
        error = vive - target
        out.append(
            {
                "trial_id": trial_id,
                "target_id": target_id,
                "settle_window_s": settle_window_s,
                "sample_count": len(settled),
                "q1_encoder_rad_mean": _mean(settled, "q1_encoder_rad"),
                "q2_encoder_rad_mean": _mean(settled, "q2_encoder_rad"),
                "p_target_base_x_m": target[0],
                "p_target_base_y_m": target[1],
                "p_target_base_z_m": target[2],
                "p_vive_settled_base_x_m": vive[0],
                "p_vive_settled_base_y_m": vive[1],
                "p_vive_settled_base_z_m": vive[2],
                "error_x_m": error[0],
                "error_y_m": error[1],
                "error_z_m": error[2],
                "error_norm_m": float(np.linalg.norm(error)),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build settled P1 dataset from live comparison log.")
    parser.add_argument("--log", default="outputs/pre_imu_p1/p1_live_comparison_log.csv")
    parser.add_argument("--output", default="outputs/pre_imu_p1/p1_settled_dataset.csv")
    parser.add_argument("--settle-window-s", type=float, default=0.5)
    args = parser.parse_args()

    with _resolve(args.log).open("r", newline="", encoding="utf-8") as f:
        log_rows = list(csv.DictReader(f))
    rows = build_dataset(log_rows, args.settle_window_s)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    print(f"settled_rows={len(rows)} settle_window_s={args.settle_window_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
