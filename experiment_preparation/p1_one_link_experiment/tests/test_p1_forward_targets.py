from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p1_forward_targets import (
    TARGET_COLUMNS,
    generate_p1_targets,
    load_p1_target_config,
    override_p1_target_config,
    write_p1_targets,
)


def test_p1_target_generation_required_columns_and_count(tmp_path):
    config = load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml")
    rows = generate_p1_targets(config)
    output = tmp_path / "p1_targets_pilot.csv"
    write_p1_targets(rows, output)

    with output.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        generated = list(reader)

    assert output.exists()
    assert reader.fieldnames == TARGET_COLUMNS
    assert len(generated) == 30


def test_p1_targets_are_forward_and_inside_pilot_azimuth():
    config = load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml")
    rows = generate_p1_targets(config)
    assert len(rows) == 30
    assert all(float(row["p_target_base_x_m"]) > 0.0 for row in rows)
    assert all(-30.0 <= float(row["azimuth_deg"]) <= 30.0 for row in rows)


def test_p1_targets_do_not_require_vive_or_imu_fields():
    config = load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml")
    rows = generate_p1_targets(config)
    forbidden = {"vive_x", "vive_y", "vive_z", "imu_x", "imu_y", "imu_z"}
    assert forbidden.isdisjoint(rows[0].keys())


def test_p1_target_count_can_be_overridden():
    config = override_p1_target_config(load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml"), target_count=64)
    rows = generate_p1_targets(config)
    assert len(rows) == 64
    assert all(float(row["p_target_base_x_m"]) > 0.0 for row in rows)


def test_p1_combined_bin_probability_is_nearly_constant():
    config = override_p1_target_config(load_p1_target_config(ROOT / "config" / "p1_forward_workspace.yaml"), target_count=75)
    rows = generate_p1_targets(config)
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['yaw_bin']}|{row['pitch_bin']}|{row['height_bin']}"
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1
