from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sampling.p2_forward_targets import REQUIRED_TARGET_COLUMNS, generate_targets, write_targets


def test_generate_targets_contract(tmp_path) -> None:
    targets, metadata = generate_targets(
        ROOT / "config" / "p2_geometry.yaml",
        ROOT / "config" / "p2_forward_workspace.yaml",
        target_count=20,
        candidate_count=1000,
        seed=7,
    )
    assert len(targets) == 20
    assert set(REQUIRED_TARGET_COLUMNS).issubset(targets.columns)
    assert (targets["p_target_base_x_m"] > 0).all()
    assert targets["azimuth_deg"].between(-30.0, 30.0).all()
    forbidden_sensor_columns = [
        col
        for col in targets.columns
        if col.lower().startswith("vive")
        or col.lower().startswith("imu")
        or "tracker" in col.lower()
    ]
    assert forbidden_sensor_columns == []
    assert metadata["geometry"]["L1_effective_m"] == 0.2996
    assert metadata["geometry"]["L2_effective_m"] == 0.29945
    assert "Vive Tracker settled position is label" in metadata["label_policy"]
    output = tmp_path / "targets.csv"
    meta = tmp_path / "targets_meta.json"
    write_targets(targets, metadata, output, meta)
    assert len(pd.read_csv(output)) == 20
    assert json.loads(meta.read_text(encoding="utf-8"))["target_count"] == 20
