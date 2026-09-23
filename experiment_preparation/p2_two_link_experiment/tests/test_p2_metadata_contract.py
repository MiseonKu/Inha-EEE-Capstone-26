from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generated_metadata_contract_if_outputs_exist() -> None:
    meta_path = ROOT / "outputs" / "pre_imu_p2" / "p2_targets_pilot_metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["stage"] == "P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL"
    assert meta["geometry"]["L1_effective_m"] == 0.2996
    assert meta["geometry"]["L2_effective_m"] == 0.29945
    assert meta["geometry"]["link1_flexible_bar_length_m"] == 0.233
    assert meta["geometry"]["link2_flexible_bar_length_m"] == 0.215
    assert meta["geometry"]["material_preset"] == "TPU95A_40x12_wall4_topbottom4_infill15_honeycomb"
    assert meta["target_generation_version"] == "p2_forward_targets_v1"
    assert meta["coordinate_frame"] == "base_link"
    assert "Vive Tracker settled position is label" in meta["label_policy"]

    for path in [
        ROOT / "outputs" / "pre_imu_p2" / "p2_trial_schedule_metadata.json",
        ROOT / "outputs" / "pre_imu_p2" / "p2_trajectories_metadata.json",
    ]:
        item = json.loads(path.read_text(encoding="utf-8"))
        assert item["stage"] == "P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL"
        assert item["geometry"]["L1_effective_m"] == 0.2996
        assert item["geometry"]["L2_effective_m"] == 0.29945
        assert item["geometry"]["link1_flexible_bar_length_m"] == 0.233
        assert item["geometry"]["link2_flexible_bar_length_m"] == 0.215
        assert item["geometry"]["material_preset"] == "TPU95A_40x12_wall4_topbottom4_infill15_honeycomb"
        assert item["coordinate_frame"] == "base_link"
        assert "label" in item["label_policy"].lower()
