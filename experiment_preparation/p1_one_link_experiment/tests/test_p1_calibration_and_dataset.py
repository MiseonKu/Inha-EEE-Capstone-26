from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_p1_settled_dataset import build_dataset
from scripts.calibrate_p1_vive_to_base import estimate_rigid_transform


def test_p1_vive_to_base_rigid_calibration_recovers_transform():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=float,
    )
    theta = np.deg2rad(30.0)
    R_true = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = np.array([0.1, -0.03, 0.2])
    target = (R_true @ source.T).T + t_true

    R, t, residual = estimate_rigid_transform(source, target)
    assert np.allclose(R, R_true, atol=1e-12)
    assert np.allclose(t, t_true, atol=1e-12)
    assert np.linalg.norm(residual) < 1e-12


def test_p1_settled_dataset_uses_last_window_samples():
    rows = []
    for i, stamp in enumerate([0.0, 0.2, 0.4, 0.8, 1.0]):
        rows.append(
            {
                "stamp_s": str(stamp),
                "trial_id": "0",
                "target_id": "2",
                "q1_encoder_rad": "0.1",
                "q2_encoder_rad": "1.2",
                "p_target_base_x_m": "0.2",
                "p_target_base_y_m": "0.0",
                "p_target_base_z_m": "0.2",
                "p_vive_base_x_m": str(0.2 + 0.01 * i),
                "p_vive_base_y_m": "0.01",
                "p_vive_base_z_m": "0.19",
            }
        )
    dataset = build_dataset(rows, settle_window_s=0.25)
    assert len(dataset) == 1
    assert dataset[0]["sample_count"] == 2
    assert np.isclose(dataset[0]["p_vive_settled_base_x_m"], (0.23 + 0.24) / 2.0)
    assert dataset[0]["error_norm_m"] > 0.0
