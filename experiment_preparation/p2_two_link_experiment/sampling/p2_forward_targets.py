"""Forward-trimmed-hemisphere target generation for AlphaF P2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kinematics.p2_fk import P2Geometry, fk_p2_batch, load_geometry, load_yaml


TARGET_GENERATION_VERSION = "p2_forward_targets_v1"
REQUIRED_TARGET_COLUMNS = [
    "target_id",
    "stage",
    "link_count",
    "q1_rad",
    "q2_rad",
    "q3_rad",
    "q1_deg",
    "q2_deg",
    "q3_deg",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "p_target_base_x_mm",
    "p_target_base_y_mm",
    "p_target_base_z_mm",
    "azimuth_deg",
    "radius_m",
    "height_m",
    "azimuth_bin",
    "radial_bin",
    "height_bin",
    "posture_bin",
    "bending_proxy_bin",
    "workspace_region",
    "L1_effective_m",
    "L2_effective_m",
    "link1_flexible_bar_length_m",
    "link2_flexible_bar_length_m",
    "material_preset",
    "target_generation_version",
]


def generate_candidates(
    geom: P2Geometry,
    workspace_cfg: dict[str, Any],
    *,
    candidate_count: int | None = None,
    seed: int | None = None,
    q_ranges_deg: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Generate joint-space candidates and FK results."""

    n = int(candidate_count or workspace_cfg["candidate_count"])
    rng = np.random.default_rng(int(seed if seed is not None else workspace_cfg["seed"]))
    ranges = q_ranges_deg or workspace_cfg["q_ranges_deg"]
    q_deg = np.column_stack(
        [
            rng.uniform(*ranges["q1"], size=n),
            rng.uniform(*ranges["q2"], size=n),
            rng.uniform(*ranges["q3"], size=n),
        ]
    )
    q_rad = np.deg2rad(q_deg)
    p = fk_p2_batch(q_rad, geom)
    center = np.asarray(workspace_cfg["center_m"], dtype=float)
    v = p - center
    azimuth = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    radius = np.linalg.norm(v, axis=1)
    bending_proxy = (
        0.45 * np.abs(np.sin(q_rad[:, 1]))
        + 0.35 * np.abs(np.sin(q_rad[:, 1] + q_rad[:, 2]))
        + 0.20 * np.abs(q_rad[:, 2]) / math.pi
    )
    df = pd.DataFrame(
        {
            "q1_rad": q_rad[:, 0],
            "q2_rad": q_rad[:, 1],
            "q3_rad": q_rad[:, 2],
            "q1_deg": q_deg[:, 0],
            "q2_deg": q_deg[:, 1],
            "q3_deg": q_deg[:, 2],
            "p_target_base_x_m": p[:, 0],
            "p_target_base_y_m": p[:, 1],
            "p_target_base_z_m": p[:, 2],
            "azimuth_deg": azimuth,
            "radius_m": radius,
            "height_m": p[:, 2],
            "bending_proxy": bending_proxy,
        }
    )
    az_min, az_max = workspace_cfg["pilot_azimuth_limit_deg"]
    df["inside_forward_region"] = (
        (df["p_target_base_x_m"] > center[0])
        & (df["azimuth_deg"] >= float(az_min))
        & (df["azimuth_deg"] <= float(az_max))
    )
    return df


def add_bins(df: pd.DataFrame, workspace_cfg: dict[str, Any]) -> pd.DataFrame:
    """Add workspace and posture bin labels to valid candidate rows."""

    out = df.copy()
    valid = out["inside_forward_region"]
    if not valid.any():
        raise ValueError("no candidates inside forward trimmed hemisphere")
    bin_cfg = workspace_cfg["bins"]
    out.loc[valid, "azimuth_bin"] = _equal_width_bin(
        out.loc[valid, "azimuth_deg"], int(bin_cfg["azimuth"]), "azimuth"
    )
    out.loc[valid, "radial_bin"] = _equal_width_bin(
        out.loc[valid, "radius_m"], int(bin_cfg["radial"]), "radial"
    )
    out.loc[valid, "height_bin"] = _equal_width_bin(
        out.loc[valid, "height_m"], int(bin_cfg["height"]), "height"
    )
    out.loc[valid, "posture_bin"] = _posture_bin(out.loc[valid, "q3_deg"])
    out.loc[valid, "bending_proxy_bin"] = _quantile_bin(
        out.loc[valid, "bending_proxy"], int(bin_cfg["bending_proxy"]), "bend"
    )
    out.loc[valid, "bin_id"] = (
        out.loc[valid, "azimuth_bin"].astype(str)
        + "|"
        + out.loc[valid, "radial_bin"].astype(str)
        + "|"
        + out.loc[valid, "height_bin"].astype(str)
        + "|"
        + out.loc[valid, "posture_bin"].astype(str)
        + "|"
        + out.loc[valid, "bending_proxy_bin"].astype(str)
    )
    return out


def _equal_width_bin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series([f"{prefix}_0"] * len(series), index=series.index)
    labels = [f"{prefix}_{i}" for i in range(n_bins)]
    return pd.cut(series, bins=n_bins, labels=labels, include_lowest=True).astype(str)


def _quantile_bin(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series([f"{prefix}_0"] * len(series), index=series.index)
    labels = [f"{prefix}_{i}" for i in range(n_bins)]
    try:
        return pd.qcut(series.rank(method="first"), q=n_bins, labels=labels).astype(str)
    except ValueError:
        return _equal_width_bin(series, n_bins, prefix)


def _posture_bin(q3_deg: pd.Series) -> pd.Series:
    bins = [-np.inf, -95.0, -55.0, np.inf]
    labels = ["folded", "neutral", "extended"]
    return pd.cut(q3_deg, bins=bins, labels=labels, include_lowest=True).astype(str)


def select_balanced_targets(df: pd.DataFrame, target_count: int, seed: int) -> pd.DataFrame:
    """Select targets with approximately uniform occupied-bin coverage."""

    valid = df[df["inside_forward_region"]].dropna(subset=["bin_id"]).copy()
    if valid.empty:
        raise ValueError("no valid binned candidates")
    rng = np.random.default_rng(seed)
    counts = valid.groupby("bin_id").size().sort_values(ascending=True)
    allocation = {bin_id: 0 for bin_id in counts.index}
    remaining = int(target_count)
    while remaining > 0:
        progressed = False
        for bin_id, available in counts.items():
            if remaining <= 0:
                break
            if allocation[bin_id] < int(available):
                allocation[bin_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    parts = []
    for bin_id, n_select in allocation.items():
        if n_select <= 0:
            continue
        group = valid[valid["bin_id"] == bin_id]
        idx = rng.choice(group.index.to_numpy(), size=n_select, replace=False)
        parts.append(group.loc[idx])
    selected = pd.concat(parts, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected.insert(0, "target_id", [f"P2_T{i:03d}" for i in range(1, len(selected) + 1)])
    return selected


def target_columns(selected: pd.DataFrame, geom: P2Geometry, workspace_cfg: dict[str, Any]) -> pd.DataFrame:
    """Project selected rows into the required target CSV schema."""

    out = selected.copy()
    out.insert(1, "stage", geom.stage)
    out.insert(2, "link_count", 2)
    out["p_target_base_x_mm"] = out["p_target_base_x_m"] * 1000.0
    out["p_target_base_y_mm"] = out["p_target_base_y_m"] * 1000.0
    out["p_target_base_z_mm"] = out["p_target_base_z_m"] * 1000.0
    out["workspace_region"] = workspace_cfg["workspace_mode"]
    out["L1_effective_m"] = geom.L1_effective_m
    out["L2_effective_m"] = geom.L2_effective_m
    out["link1_flexible_bar_length_m"] = geom.link1_flexible_bar_length_m
    out["link2_flexible_bar_length_m"] = geom.link2_flexible_bar_length_m
    out["material_preset"] = geom.material_preset
    out["target_generation_version"] = TARGET_GENERATION_VERSION
    return out[REQUIRED_TARGET_COLUMNS]


def generate_targets(
    geometry_path: str | Path,
    workspace_path: str | Path,
    *,
    target_count: int | None = None,
    candidate_count: int | None = None,
    seed: int | None = None,
    q_ranges_deg: dict[str, list[float]] | None = None,
    azimuth_range_deg: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate balanced P2 pilot targets and metadata."""

    geom = load_geometry(geometry_path)
    workspace_cfg = load_yaml(workspace_path)
    if azimuth_range_deg is not None:
        workspace_cfg = dict(workspace_cfg)
        workspace_cfg["pilot_azimuth_limit_deg"] = [float(azimuth_range_deg[0]), float(azimuth_range_deg[1])]
    target_n = int(target_count or workspace_cfg["pilot_target_count"])
    seed_value = int(seed if seed is not None else workspace_cfg["seed"])
    candidates = generate_candidates(
        geom,
        workspace_cfg,
        candidate_count=candidate_count,
        seed=seed_value,
        q_ranges_deg=q_ranges_deg,
    )
    binned = add_bins(candidates, workspace_cfg)
    selected = select_balanced_targets(binned, target_n, seed_value)
    targets = target_columns(selected, geom, workspace_cfg)
    metadata = build_target_metadata(targets, binned, geom, workspace_cfg, seed_value, int(candidate_count or workspace_cfg["candidate_count"]))
    return targets, metadata


def build_target_metadata(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    geom: P2Geometry,
    workspace_cfg: dict[str, Any],
    seed: int,
    candidate_count: int,
) -> dict[str, Any]:
    selected_counts = targets.groupby(["azimuth_bin", "radial_bin", "height_bin", "posture_bin", "bending_proxy_bin"]).size()
    candidate_counts = candidates[candidates["inside_forward_region"]].groupby("bin_id").size()
    values = selected_counts.to_numpy(dtype=int)
    return {
        "stage": geom.stage,
        "coordinate_frame": geom.base_frame,
        "label_policy": "target is rigid reference coordinate; Vive Tracker settled position is label; Vive is not control feedback; IMU is not used",
        "target_generation_version": TARGET_GENERATION_VERSION,
        "seed": seed,
        "target_count": int(len(targets)),
        "candidate_count": int(candidate_count),
        "valid_candidate_count": int(candidates["inside_forward_region"].sum()),
        "occupied_bin_count": int(candidate_counts.size),
        "selected_bin_min": int(values.min()) if len(values) else 0,
        "selected_bin_max": int(values.max()) if len(values) else 0,
        "selected_bin_max_minus_min": int(values.max() - values.min()) if len(values) else 0,
        "q_ranges_deg": workspace_cfg["q_ranges_deg"],
        "azimuth_limit_deg": workspace_cfg["pilot_azimuth_limit_deg"],
        "geometry": {
            "L1_effective_m": geom.L1_effective_m,
            "L2_effective_m": geom.L2_effective_m,
            "link1_flexible_bar_length_m": geom.link1_flexible_bar_length_m,
            "link2_flexible_bar_length_m": geom.link2_flexible_bar_length_m,
            "link1_nonflex_offset_mm": geom.link1_nonflex_offset_mm,
            "link2_nonflex_offset_mm": geom.link2_nonflex_offset_mm,
            "material_preset": geom.material_preset,
        },
        "bin_count_tables": {
            "selected": {"|".join(map(str, key)): int(value) for key, value in selected_counts.items()},
            "candidate_occupied": {str(key): int(value) for key, value in candidate_counts.items()},
        },
        "csv_sha256": dataframe_digest(targets),
    }


def dataframe_digest(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_targets(targets: pd.DataFrame, metadata: dict[str, Any], output: str | Path, metadata_output: str | Path) -> None:
    """Write target CSV and metadata JSON."""

    output = Path(output)
    metadata_output = Path(metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output, index=False)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
