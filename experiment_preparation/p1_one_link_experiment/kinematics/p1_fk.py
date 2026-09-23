"""Rigid two-joint FK for the AlphaF P1 one-link setup."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

import numpy as np


L1_EFFECTIVE_M = 0.274123
L1_EFFECTIVE_MM = 274.123
FLEXIBLE_BAR_LENGTH_M = 0.233
FLEXIBLE_BAR_LENGTH_MM = 233.0
ADAPTER_OFFSET_M = 0.041123
ADAPTER_OFFSET_MM = 41.123
MATERIAL_PRESET = "TPU95A_40x12_wall4_topbottom4_infill15_honeycomb"
STAGE = "P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL"
Q_BASIC = np.array([0.0, pi / 2.0], dtype=float)


@dataclass(frozen=True)
class P1FKCase:
    """One deterministic FK validation case."""

    name: str
    q1_rad: float
    q2_rad: float
    expected_m: np.ndarray


def fk_p1(q: Iterable[float] | np.ndarray, L1_m: float = L1_EFFECTIVE_M) -> np.ndarray:
    """Return the rigid P1 tool position in base_link meters.

    The convention matches the AlphaF P1 experiment definition:

    rho = L1 * cos(q2)
    z   = L1 * sin(q2)
    x   = rho * cos(q1)
    y   = rho * sin(q1)
    """

    q_arr = np.asarray(q, dtype=float).reshape(2)
    q1, q2 = float(q_arr[0]), float(q_arr[1])
    rho = L1_m * cos(q2)
    return np.array([rho * cos(q1), rho * sin(q1), L1_m * sin(q2)], dtype=float)


def p1_validation_cases(L1_m: float = L1_EFFECTIVE_M) -> list[P1FKCase]:
    """Return the required FK validation cases."""

    return [
        P1FKCase("forward_horizontal", 0.0, 0.0, np.array([L1_m, 0.0, 0.0], dtype=float)),
        P1FKCase("vertical_basic", 0.0, pi / 2.0, np.array([0.0, 0.0, L1_m], dtype=float)),
        P1FKCase("left_horizontal", pi / 2.0, 0.0, np.array([0.0, L1_m, 0.0], dtype=float)),
    ]


def validate_p1_fk(atol: float = 1e-9, L1_m: float = L1_EFFECTIVE_M) -> list[dict[str, object]]:
    """Run the required FK validation cases and return structured results."""

    results: list[dict[str, object]] = []
    for case in p1_validation_cases(L1_m):
        actual = fk_p1([case.q1_rad, case.q2_rad], L1_m=L1_m)
        error = float(np.linalg.norm(actual - case.expected_m))
        results.append(
            {
                "name": case.name,
                "q1_rad": case.q1_rad,
                "q2_rad": case.q2_rad,
                "actual_m": actual,
                "expected_m": case.expected_m,
                "error_m": error,
                "pass": bool(np.allclose(actual, case.expected_m, atol=atol)),
            }
        )
    return results
