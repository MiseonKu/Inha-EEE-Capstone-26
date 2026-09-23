#!/usr/bin/env python3
"""Estimate Vive-world to AlphaF base_link rigid transform from paired points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_COLUMNS = ["p_base_x_m", "p_base_y_m", "p_base_z_m"]
TARGET_COLUMNS = ["p_target_base_x_m", "p_target_base_y_m", "p_target_base_z_m"]
VIVE_COLUMNS = ["p_vive_world_x_m", "p_vive_world_y_m", "p_vive_world_z_m"]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _read_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError("at least three non-collinear calibration pairs are required")
    base_cols = BASE_COLUMNS if set(BASE_COLUMNS).issubset(rows[0]) else TARGET_COLUMNS
    if not set(base_cols).issubset(rows[0]) or not set(VIVE_COLUMNS).issubset(rows[0]):
        raise ValueError(f"CSV must contain {base_cols} and {VIVE_COLUMNS}")
    p_base = np.array([[float(row[col]) for col in base_cols] for row in rows], dtype=float)
    p_vive = np.array([[float(row[col]) for col in VIVE_COLUMNS] for row in rows], dtype=float)
    return p_vive, p_base


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate R,t so target ~= R @ source + t."""

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid
    residual = (R @ source.T).T + t - target
    return R, t, residual


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Vive world coordinates into base_link coordinates.")
    parser.add_argument("--pairs", required=True, help="CSV with paired base and Vive points.")
    parser.add_argument("--output", default="outputs/pre_imu_p1/p1_vive_to_base_calibration.json")
    args = parser.parse_args()

    p_vive, p_base = _read_pairs(_resolve(args.pairs))
    R, t, residual = estimate_rigid_transform(p_vive, p_base)
    residual_norm = np.linalg.norm(residual, axis=1)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frame": "vive_world",
        "target_frame": "base_link",
        "R_base_from_vive_world": R.tolist(),
        "t_base_from_vive_world_m": t.tolist(),
        "pair_count": int(len(p_vive)),
        "rmse_m": float(np.sqrt(np.mean(residual_norm**2))),
        "max_error_m": float(residual_norm.max()),
        "mean_error_m": float(residual_norm.mean()),
        "input_pairs": str(_resolve(args.pairs)),
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"wrote {output}")
    print(f"pairs={len(p_vive)} rmse_m={data['rmse_m']:.6f} max_error_m={data['max_error_m']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
