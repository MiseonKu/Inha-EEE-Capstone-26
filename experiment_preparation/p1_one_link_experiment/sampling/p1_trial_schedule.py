"""Balanced P1 trial schedule generation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEDULE_COLUMNS = [
    "trial_id",
    "target_id",
    "repeat_index",
    "cycle_id",
    "planned_order_index",
    "q1_rad",
    "q2_rad",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "yaw_bin",
    "pitch_bin",
    "height_bin",
    "workspace_region",
]


def read_targets(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["target_id"]:
            row[key] = int(row[key])
        for key in ["q1_rad", "q2_rad", "p_target_base_x_m", "p_target_base_y_m", "p_target_base_z_m"]:
            row[key] = float(row[key])
    return rows


def _interleaved_target_ids(targets: list[dict[str, Any]], rng: np.random.Generator) -> list[int]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(targets):
        bin_id = f"{row.get('yaw_bin', '')}|{row.get('pitch_bin', '')}|{row.get('height_bin', '')}"
        groups.setdefault(bin_id, []).append(idx)
    bins = list(groups.keys())
    rng.shuffle(bins)
    for indices in groups.values():
        rng.shuffle(indices)

    order: list[int] = []
    max_len = max(len(v) for v in groups.values())
    for layer in range(max_len):
        active = [bin_id for bin_id in bins if layer < len(groups[bin_id])]
        rng.shuffle(active)
        for bin_id in active:
            order.append(groups[bin_id][layer])
    return order


def _repair_cycle_boundary(order: list[int], targets: list[dict[str, Any]], previous_target_id: int | None) -> list[int]:
    if previous_target_id is None or not order:
        return order
    if int(targets[order[0]]["target_id"]) != previous_target_id:
        return order
    for idx in range(1, len(order)):
        if int(targets[order[idx]]["target_id"]) != previous_target_id:
            order[0], order[idx] = order[idx], order[0]
            break
    return order


def generate_p1_schedule(targets: list[dict[str, Any]], repeats: int = 3, seed: int = 202) -> list[dict[str, Any]]:
    """Generate a repeated, bin-interleaved schedule."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    schedule: list[dict[str, Any]] = []
    previous_target_id: int | None = None
    trial_id = 0
    for repeat_index in range(repeats):
        order = _interleaved_target_ids(targets, rng)
        order = _repair_cycle_boundary(order, targets, previous_target_id)
        for idx in order:
            target = targets[idx]
            row = {
                "trial_id": trial_id,
                "target_id": int(target["target_id"]),
                "repeat_index": repeat_index,
                "cycle_id": repeat_index,
                "planned_order_index": trial_id,
                "q1_rad": float(target["q1_rad"]),
                "q2_rad": float(target["q2_rad"]),
                "p_target_base_x_m": float(target["p_target_base_x_m"]),
                "p_target_base_y_m": float(target["p_target_base_y_m"]),
                "p_target_base_z_m": float(target["p_target_base_z_m"]),
                "yaw_bin": target.get("yaw_bin", ""),
                "pitch_bin": target.get("pitch_bin", ""),
                "height_bin": target.get("height_bin", ""),
                "workspace_region": target.get("workspace_region", ""),
            }
            schedule.append(row)
            previous_target_id = row["target_id"]
            trial_id += 1
    return schedule


def write_p1_schedule(rows: list[dict[str, Any]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEDULE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in SCHEDULE_COLUMNS})


def write_p1_schedule_metadata(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    repeats: int,
    seed: int,
    output: str | Path,
) -> None:
    adjacent_same = 0
    adjacent_same_bin = 0
    for prev, cur in zip(rows, rows[1:]):
        if prev["target_id"] == cur["target_id"]:
            adjacent_same += 1
        prev_bin = (prev.get("yaw_bin"), prev.get("pitch_bin"), prev.get("height_bin"))
        cur_bin = (cur.get("yaw_bin"), cur.get("pitch_bin"), cur.get("height_bin"))
        if prev_bin == cur_bin:
            adjacent_same_bin += 1
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_count": len(targets),
        "repeats": repeats,
        "seed": seed,
        "trial_count": len(rows),
        "adjacent_same_target_count": adjacent_same,
        "adjacent_same_bin_count": adjacent_same_bin,
        "policy": "Each cycle contains every target once; bins are interleaved and cycle boundaries are repaired when possible.",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
