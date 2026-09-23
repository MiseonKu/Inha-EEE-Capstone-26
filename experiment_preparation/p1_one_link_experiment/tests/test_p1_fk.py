from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p1_fk import L1_EFFECTIVE_M, fk_p1, validate_p1_fk


def test_p1_fk_required_cases_pass():
    results = validate_p1_fk()
    assert all(result["pass"] for result in results)


def test_p1_fk_forward_horizontal():
    assert np.allclose(fk_p1([0.0, 0.0]), [L1_EFFECTIVE_M, 0.0, 0.0], atol=1e-12)


def test_p1_fk_vertical_basic_pose():
    assert np.allclose(fk_p1([0.0, np.pi / 2.0]), [0.0, 0.0, L1_EFFECTIVE_M], atol=1e-12)


def test_p1_fk_left_horizontal():
    assert np.allclose(fk_p1([np.pi / 2.0, 0.0]), [0.0, L1_EFFECTIVE_M, 0.0], atol=1e-12)
