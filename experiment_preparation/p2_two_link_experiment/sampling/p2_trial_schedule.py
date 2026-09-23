"""Repeat-balanced P2 trial schedule generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEDULE_COLUMNS = [
    "trial_id",
    "planned_order_index",
    "cycle_id",
    "target_id",
    "repeat_index",
    "q1_rad",
    "q2_rad",
    "q3_rad",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "azimuth_bin",
    "radial_bin",
    "height_bin",
    "posture_bin",
]


def generate_schedule(targets: pd.DataFrame, repeats: int = 3, seed: int = 303) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a repeat-balanced schedule with cycle-wise shuffling."""

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = np.random.default_rng(seed)
    rows: list[pd.Series] = []
    previous_target = None
    for cycle_id in range(1, repeats + 1):
        shuffled = _interleave_bins(targets, rng)
        if previous_target is not None and len(shuffled) > 1 and shuffled.iloc[0]["target_id"] == previous_target:
            shuffled = _swap_first_nonmatching(shuffled, previous_target)
        for _, row in shuffled.iterrows():
            out = row.copy()
            out["cycle_id"] = cycle_id
            out["repeat_index"] = cycle_id
            rows.append(out)
            previous_target = row["target_id"]
    schedule = pd.DataFrame(rows).reset_index(drop=True)
    schedule.insert(0, "trial_id", [f"P2_TRIAL_{i:04d}" for i in range(1, len(schedule) + 1)])
    schedule.insert(1, "planned_order_index", np.arange(len(schedule), dtype=int))
    schedule = schedule[SCHEDULE_COLUMNS]
    metadata = {
        "stage": targets["stage"].iloc[0] if "stage" in targets else "P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL",
        "coordinate_frame": "base_link",
        "seed": seed,
        "repeats": repeats,
        "target_count": int(targets["target_id"].nunique()),
        "trial_count": int(len(schedule)),
        "all_targets_expected_repeats": bool(schedule.groupby("target_id").size().eq(repeats).all()),
        "adjacent_same_target_count": int((schedule["target_id"].shift() == schedule["target_id"]).sum()),
        "geometry": {
            "L1_effective_m": float(targets["L1_effective_m"].iloc[0]) if "L1_effective_m" in targets else None,
            "L2_effective_m": float(targets["L2_effective_m"].iloc[0]) if "L2_effective_m" in targets else None,
            "link1_flexible_bar_length_m": float(targets["link1_flexible_bar_length_m"].iloc[0]) if "link1_flexible_bar_length_m" in targets else None,
            "link2_flexible_bar_length_m": float(targets["link2_flexible_bar_length_m"].iloc[0]) if "link2_flexible_bar_length_m" in targets else None,
            "material_preset": str(targets["material_preset"].iloc[0]) if "material_preset" in targets else None,
        },
        "target_generation_version": str(targets["target_generation_version"].iloc[0]) if "target_generation_version" in targets else None,
        "label_policy": "schedule contains rigid target references only; Vive Tracker is label/evaluation only",
        "csv_sha256": dataframe_digest(schedule),
    }
    return schedule, metadata


def _interleave_bins(targets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = []
    key_cols = ["azimuth_bin", "radial_bin", "height_bin", "posture_bin"]
    for _, group in targets.groupby(key_cols, dropna=False):
        groups.append(group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True))
    rng.shuffle(groups)
    rows = []
    max_len = max(len(group) for group in groups)
    for i in range(max_len):
        for group in groups:
            if i < len(group):
                rows.append(group.iloc[i])
    return pd.DataFrame(rows).reset_index(drop=True)


def _swap_first_nonmatching(df: pd.DataFrame, previous_target: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for idx in range(1, len(out)):
        if out.loc[idx, "target_id"] != previous_target:
            first = out.iloc[0].copy()
            out.iloc[0] = out.iloc[idx]
            out.iloc[idx] = first
            break
    return out


def dataframe_digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


def write_schedule(schedule: pd.DataFrame, metadata: dict[str, Any], output: str | Path, metadata_output: str | Path) -> None:
    output = Path(output)
    metadata_output = Path(metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
