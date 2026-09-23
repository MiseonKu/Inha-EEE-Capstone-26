from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p2_fk import check_validation_cases, fk_p2, load_geometry


def test_p2_fk_validation_cases() -> None:
    geom = load_geometry(ROOT / "config" / "p2_geometry.yaml")
    rows = check_validation_cases(geom)
    assert all(row["passed"] for row in rows)


def test_p2_geometry_constants() -> None:
    geom = load_geometry(ROOT / "config" / "p2_geometry.yaml")
    assert math.isclose(geom.L1_effective_m, 0.2996)
    assert math.isclose(geom.L2_effective_m, 0.29945)
    assert math.isclose(geom.link1_nonflex_offset_mm, 66.6)
    assert math.isclose(geom.link2_nonflex_offset_mm, 84.45)
    assert geom.material_preset == "TPU95A_40x12_wall4_topbottom4_infill15_honeycomb"


def test_basic_pose_is_folded_l_shape() -> None:
    geom = load_geometry(ROOT / "config" / "p2_geometry.yaml")
    p = fk_p2(geom.q_basic_rad, geom)
    assert np.allclose(p, np.array([0.29945, 0.0, 0.2996]), atol=1e-9)
