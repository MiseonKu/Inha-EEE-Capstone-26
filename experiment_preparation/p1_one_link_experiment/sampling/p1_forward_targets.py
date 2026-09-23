"""P1 forward trimmed hemisphere target generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import atan2, degrees, pi, radians, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kinematics.p1_fk import (
    L1_EFFECTIVE_M,
    L1_EFFECTIVE_MM,
    MATERIAL_PRESET,
    STAGE,
    fk_p1,
)


TARGET_COLUMNS = [
    "target_id",
    "stage",
    "link_count",
    "q1_rad",
    "q2_rad",
    "q1_deg",
    "q2_deg",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "p_target_base_x_mm",
    "p_target_base_y_mm",
    "p_target_base_z_mm",
    "azimuth_deg",
    "radius_m",
    "height_m",
    "yaw_bin",
    "pitch_bin",
    "height_bin",
    "workspace_region",
    "L1_effective_m",
    "material_preset",
]


@dataclass(frozen=True)
class P1TargetConfig:
    """Parsed P1 target generation config."""

    stage: str
    workspace_mode: str
    target_count: int
    repeats: int
    seed: int
    candidate_count: int
    q1_range_deg: tuple[float, float]
    q2_range_deg: tuple[float, float]
    azimuth_range_deg: tuple[float, float]
    L1_effective_m: float
    material_preset: str
    yaw_bins: int
    pitch_bins: int
    height_bins: int


def override_p1_target_config(config: P1TargetConfig, **kwargs: Any) -> P1TargetConfig:
    """Return a config copy with CLI/runtime overrides applied."""

    clean = {key: value for key, value in kwargs.items() if value is not None}
    for range_key in ["q1_range_deg", "q2_range_deg", "azimuth_range_deg"]:
        if range_key in clean:
            clean[range_key] = _range_pair(clean[range_key])
    return replace(config, **clean)


def load_p1_target_config(path: str | Path) -> P1TargetConfig:
    """Load target-generation settings from YAML."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    geometry = data.get("geometry", {})
    sampling = data.get("sampling", {})
    workspace = data.get("workspace", {})
    bins = data.get("bins", {})

    return P1TargetConfig(
        stage=str(data.get("stage", STAGE)),
        workspace_mode=str(workspace.get("mode", "forward_trimmed_hemisphere")),
        target_count=int(sampling.get("target_count", 30)),
        repeats=int(sampling.get("repeats", 3)),
        seed=int(sampling.get("seed", 101)),
        candidate_count=int(sampling.get("candidate_count", 5000)),
        q1_range_deg=_range_pair(sampling.get("q1_range_deg", [-30.0, 30.0])),
        q2_range_deg=_range_pair(sampling.get("q2_range_deg", [45.0, 85.0])),
        azimuth_range_deg=_range_pair(workspace.get("pilot_azimuth_deg", [-30.0, 30.0])),
        L1_effective_m=float(geometry.get("L1_effective_m", L1_EFFECTIVE_M)),
        material_preset=str(data.get("material_preset", MATERIAL_PRESET)),
        yaw_bins=int(bins.get("yaw", 3)),
        pitch_bins=int(bins.get("pitch", 5)),
        height_bins=int(bins.get("height", 5)),
    )


def _range_pair(value: Any) -> tuple[float, float]:
    low, high = value
    low_f = float(low)
    high_f = float(high)
    if high_f <= low_f:
        raise ValueError(f"invalid range: {value}")
    return low_f, high_f


def _bin_label(value: float, low: float, high: float, n_bins: int, prefix: str) -> str:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    unit = (value - low) / (high - low)
    idx = int(np.clip(np.floor(unit * n_bins), 0, n_bins - 1))
    return f"{prefix}{idx:02d}"


def _effective_candidate_count(config: P1TargetConfig) -> int:
    occupied_bin_guess = max(1, config.yaw_bins * config.pitch_bins)
    return max(config.candidate_count, config.target_count * 60, occupied_bin_guess * 120)


def _candidate_rows(config: P1TargetConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    q1_low, q1_high = config.q1_range_deg
    q2_low, q2_high = config.q2_range_deg
    az_low, az_high = config.azimuth_range_deg

    rows: list[dict[str, Any]] = []
    for _ in range(_effective_candidate_count(config)):
        q1_deg = float(rng.uniform(q1_low, q1_high))
        q2_deg = float(rng.uniform(q2_low, q2_high))
        q1 = radians(q1_deg)
        q2 = radians(q2_deg)
        p = fk_p1([q1, q2], L1_m=config.L1_effective_m)
        azimuth_deg = degrees(atan2(float(p[1]), float(p[0])))
        forward = float(p[0]) > 0.0
        az_ok = az_low <= azimuth_deg <= az_high
        if not forward or not az_ok:
            continue
        height = float(p[2])
        radius = sqrt(float(p[0]) ** 2 + float(p[1]) ** 2)
        row = {
            "stage": config.stage,
            "link_count": 1,
            "q1_rad": q1,
            "q2_rad": q2,
            "q1_deg": q1_deg,
            "q2_deg": q2_deg,
            "p_target_base_x_m": float(p[0]),
            "p_target_base_y_m": float(p[1]),
            "p_target_base_z_m": float(p[2]),
            "p_target_base_x_mm": float(p[0]) * 1000.0,
            "p_target_base_y_mm": float(p[1]) * 1000.0,
            "p_target_base_z_mm": float(p[2]) * 1000.0,
            "azimuth_deg": azimuth_deg,
            "radius_m": radius,
            "height_m": height,
            "yaw_bin": _bin_label(q1_deg, q1_low, q1_high, config.yaw_bins, "yaw_"),
            "pitch_bin": _bin_label(q2_deg, q2_low, q2_high, config.pitch_bins, "pitch_"),
            "height_bin": _bin_label(height, config.L1_effective_m * np.sin(radians(q2_low)), config.L1_effective_m * np.sin(radians(q2_high)), config.height_bins, "height_"),
            "workspace_region": config.workspace_mode,
            "L1_effective_m": config.L1_effective_m,
            "material_preset": config.material_preset,
        }
        row["_bin_id"] = f"{row['yaw_bin']}|{row['pitch_bin']}|{row['height_bin']}"
        rows.append(row)
    if len(rows) < config.target_count:
        raise ValueError(f"only generated {len(rows)} valid candidates for {config.target_count} targets")
    return rows


def _feature_vector(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(row["p_target_base_x_m"]),
            float(row["p_target_base_y_m"]),
            float(row["p_target_base_z_m"]),
            float(row["q1_rad"]),
            float(row["q2_rad"]),
            float(row["azimuth_deg"]) / 180.0,
        ],
        dtype=float,
    )


def _spread_select(rows: list[dict[str, Any]], n_select: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if n_select <= 0:
        return []
    if n_select >= len(rows):
        selected = list(rows)
        rng.shuffle(selected)
        return selected

    features = np.vstack([_feature_vector(row) for row in rows])
    lo = features.min(axis=0)
    span = np.maximum(features.max(axis=0) - lo, 1e-12)
    normalized = (features - lo) / span
    first = int(rng.integers(0, len(rows)))
    selected_idx = [first]
    min_dist = np.linalg.norm(normalized - normalized[first], axis=1)
    while len(selected_idx) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected_idx.append(next_idx)
        dist = np.linalg.norm(normalized - normalized[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return [rows[idx] for idx in selected_idx]


def _allocate_equal_bin_counts(groups: dict[str, list[dict[str, Any]]], n_targets: int, seed: int) -> dict[str, int]:
    """Allocate target counts so selected-bin probability is as flat as possible."""

    if n_targets <= 0:
        raise ValueError("target_count must be positive")
    if not groups:
        raise ValueError("no occupied bins available")

    rng = np.random.default_rng(seed + 43)
    bins = sorted(groups)
    active_count = min(n_targets, len(bins))
    if active_count < len(bins):
        bins = list(rng.choice(np.asarray(bins, dtype=object), size=active_count, replace=False))
        bins.sort()

    allocation = {bin_id: n_targets // active_count for bin_id in bins}
    remainder = n_targets - sum(allocation.values())
    shuffled = list(bins)
    rng.shuffle(shuffled)
    for bin_id in shuffled[:remainder]:
        allocation[bin_id] += 1

    overflow = 0
    for bin_id in list(allocation):
        capacity = len(groups[bin_id])
        if allocation[bin_id] > capacity:
            overflow += allocation[bin_id] - capacity
            allocation[bin_id] = capacity
    while overflow > 0:
        progressed = False
        candidates = [bin_id for bin_id in bins if allocation[bin_id] < len(groups[bin_id])]
        rng.shuffle(candidates)
        for bin_id in candidates:
            if overflow <= 0:
                break
            allocation[bin_id] += 1
            overflow -= 1
            progressed = True
        if not progressed:
            break

    if sum(allocation.values()) != n_targets:
        raise ValueError(f"could only allocate {sum(allocation.values())} targets across occupied bins")
    return {bin_id: count for bin_id, count in sorted(allocation.items()) if count > 0}


def _balanced_select(candidates: list[dict[str, Any]], n_targets: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 17)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row["_bin_id"]), []).append(row)
    for rows in groups.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    allocation = _allocate_equal_bin_counts(groups, n_targets, seed)
    for bin_id, n_select in allocation.items():
        selected.extend(_spread_select(groups[bin_id], n_select, rng))
    if len(selected) != n_targets:
        raise ValueError(f"selected {len(selected)} targets, expected {n_targets}")
    selected.sort(key=lambda row: (row["yaw_bin"], row["pitch_bin"], row["height_bin"], row["q1_deg"], row["q2_deg"]))
    for target_id, row in enumerate(selected):
        row["target_id"] = target_id
        row.pop("_bin_id", None)
    return selected


def generate_p1_targets(config: P1TargetConfig) -> list[dict[str, Any]]:
    """Generate balanced P1 pilot target rows."""

    return _balanced_select(_candidate_rows(config), config.target_count, config.seed)


def write_p1_targets(rows: list[dict[str, Any]], output: str | Path) -> None:
    """Write P1 target rows to CSV."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in TARGET_COLUMNS})


def write_p1_target_metadata(config: P1TargetConfig, rows: list[dict[str, Any]], output: str | Path) -> None:
    """Write metadata for generated P1 targets."""

    yaw_counts: dict[str, int] = {}
    pitch_counts: dict[str, int] = {}
    height_counts: dict[str, int] = {}
    combined_counts: dict[str, int] = {}
    for row in rows:
        yaw_counts[row["yaw_bin"]] = yaw_counts.get(row["yaw_bin"], 0) + 1
        pitch_counts[row["pitch_bin"]] = pitch_counts.get(row["pitch_bin"], 0) + 1
        height_counts[row["height_bin"]] = height_counts.get(row["height_bin"], 0) + 1
        combined = f"{row['yaw_bin']}|{row['pitch_bin']}|{row['height_bin']}"
        combined_counts[combined] = combined_counts.get(combined, 0) + 1
    combined_values = list(combined_counts.values())
    selected_bin_min = min(combined_values) if combined_values else 0
    selected_bin_max = max(combined_values) if combined_values else 0

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": config.stage,
        "workspace_mode": config.workspace_mode,
        "target_count": len(rows),
        "requested_target_count": config.target_count,
        "repeats": config.repeats,
        "seed": config.seed,
        "configured_candidate_count": config.candidate_count,
        "effective_candidate_count": _effective_candidate_count(config),
        "q1_range_deg": list(config.q1_range_deg),
        "q2_range_deg": list(config.q2_range_deg),
        "pilot_azimuth_deg": list(config.azimuth_range_deg),
        "forward_condition": "p_target_base_x_m > 0",
        "L1_effective_m": config.L1_effective_m,
        "L1_effective_mm": L1_EFFECTIVE_MM,
        "material_preset": config.material_preset,
        "bin_counts": {
            "yaw_bin": dict(sorted(yaw_counts.items())),
            "pitch_bin": dict(sorted(pitch_counts.items())),
            "height_bin": dict(sorted(height_counts.items())),
            "combined_bin": dict(sorted(combined_counts.items())),
        },
        "uniformity": {
            "selected_combined_bin_count": len(combined_counts),
            "selected_combined_bin_min": selected_bin_min,
            "selected_combined_bin_max": selected_bin_max,
            "selected_combined_bin_max_minus_min": selected_bin_max - selected_bin_min,
            "policy": "Allocate nearly equal counts over occupied yaw|pitch|height bins, then spread-select within each bin.",
        },
        "label_policy": "Targets are rigid reference coordinates; Vive Tracker settled positions are labels during the real experiment.",
        "excluded_sensors": ["IMU"],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
