"""Rigid FK for the AlphaF P2 two-link pre-IMU experiment.

Units are meters and radians unless a field name explicitly says otherwise.
The STEP file is visual reference only; the values in ``p2_geometry.yaml`` are
the kinematic authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY_PATH = ROOT / "config" / "p2_geometry.yaml"


@dataclass(frozen=True)
class P2Geometry:
    """Authoritative P2 two-link geometry and metadata."""

    stage: str
    robot_name: str
    base_frame: str
    tool_frame: str
    tip_frame: str
    L1_effective_m: float
    L2_effective_m: float
    L1_effective_mm: float
    L2_effective_mm: float
    link1_flexible_bar_length_m: float
    link2_flexible_bar_length_m: float
    link1_flexible_bar_length_mm: float
    link2_flexible_bar_length_mm: float
    link1_nonflex_offset_m: float
    link2_nonflex_offset_m: float
    link1_nonflex_offset_mm: float
    link2_nonflex_offset_mm: float
    cross_section_mm: tuple[float, float]
    material_preset: str
    rigid_reference_only: bool
    simulate_flexible_bending: bool
    raw: dict[str, Any]

    @property
    def nominal_reach_m(self) -> float:
        """Nominal fully extended reach in meters."""

        return self.L1_effective_m + self.L2_effective_m

    @property
    def q_basic_rad(self) -> tuple[float, float, float]:
        """Nominal folded L-shaped P2 basic pose."""

        return (0.0, math.pi / 2.0, -math.pi / 2.0)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def load_geometry(path: str | Path = DEFAULT_GEOMETRY_PATH) -> P2Geometry:
    """Load authoritative P2 geometry from YAML."""

    data = load_yaml(path)
    geom = P2Geometry(
        stage=str(data["stage"]),
        robot_name=str(data["robot_name"]),
        base_frame=str(data["base_frame"]),
        tool_frame=str(data["tool_frame"]),
        tip_frame=str(data["tip_frame"]),
        L1_effective_m=float(data["L1_effective_m"]),
        L2_effective_m=float(data["L2_effective_m"]),
        L1_effective_mm=float(data["L1_effective_mm"]),
        L2_effective_mm=float(data["L2_effective_mm"]),
        link1_flexible_bar_length_m=float(data["link1_flexible_bar_length_m"]),
        link2_flexible_bar_length_m=float(data["link2_flexible_bar_length_m"]),
        link1_flexible_bar_length_mm=float(data["link1_flexible_bar_length_mm"]),
        link2_flexible_bar_length_mm=float(data["link2_flexible_bar_length_mm"]),
        link1_nonflex_offset_m=float(data["link1_nonflex_offset_m"]),
        link2_nonflex_offset_m=float(data["link2_nonflex_offset_m"]),
        link1_nonflex_offset_mm=float(data["link1_nonflex_offset_mm"]),
        link2_nonflex_offset_mm=float(data["link2_nonflex_offset_mm"]),
        cross_section_mm=tuple(float(v) for v in data["cross_section_mm"]),
        material_preset=str(data["material_preset"]),
        rigid_reference_only=bool(data["rigid_reference_only"]),
        simulate_flexible_bending=bool(data["simulate_flexible_bending"]),
        raw=data,
    )
    _validate_geometry(geom)
    return geom


def _validate_geometry(geom: P2Geometry) -> None:
    expected_l1_offset = geom.L1_effective_mm - geom.link1_flexible_bar_length_mm
    expected_l2_offset = geom.L2_effective_mm - geom.link2_flexible_bar_length_mm
    if not math.isclose(expected_l1_offset, geom.link1_nonflex_offset_mm, abs_tol=1e-9):
        raise ValueError("link1_nonflex_offset_mm does not match L1_effective_mm - bar length")
    if not math.isclose(expected_l2_offset, geom.link2_nonflex_offset_mm, abs_tol=1e-9):
        raise ValueError("link2_nonflex_offset_mm does not match L2_effective_mm - bar length")
    if geom.L1_effective_m <= 0 or geom.L2_effective_m <= 0:
        raise ValueError("effective link lengths must be positive")
    if geom.simulate_flexible_bending:
        raise ValueError("P2 package is rigid-reference only; flexible bending simulation must remain false")


def as_q(q: Iterable[float]) -> np.ndarray:
    """Return q as a length-3 float vector in radians."""

    arr = np.asarray(list(q), dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("q contains non-finite values")
    return arr


def fk_p2(q: Iterable[float], geom: P2Geometry | None = None) -> np.ndarray:
    """Compute rigid FK for q=[q1,q2,q3], returning tool0 xyz in meters."""

    geom = geom or load_geometry()
    q1, q2, q3 = as_q(q)
    L1 = geom.L1_effective_m
    L2 = geom.L2_effective_m
    rho = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
    z = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
    x = rho * math.cos(q1)
    y = rho * math.sin(q1)
    return np.array([x, y, z], dtype=float)


def fk_p2_batch(q: np.ndarray, geom: P2Geometry) -> np.ndarray:
    """Vectorized rigid FK for an Nx3 q array."""

    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (N, 3)")
    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]
    rho = geom.L1_effective_m * np.cos(q2) + geom.L2_effective_m * np.cos(q2 + q3)
    z = geom.L1_effective_m * np.sin(q2) + geom.L2_effective_m * np.sin(q2 + q3)
    x = rho * np.cos(q1)
    y = rho * np.sin(q1)
    return np.column_stack([x, y, z])


def validation_cases(geom: P2Geometry | None = None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return required FK validation cases."""

    geom = geom or load_geometry()
    L1 = geom.L1_effective_m
    L2 = geom.L2_effective_m
    return [
        ("extended_forward", np.array([0.0, 0.0, 0.0]), np.array([L1 + L2, 0.0, 0.0])),
        ("vertical_stack", np.array([0.0, math.pi / 2.0, 0.0]), np.array([0.0, 0.0, L1 + L2])),
        ("yaw_left_extended", np.array([math.pi / 2.0, 0.0, 0.0]), np.array([0.0, L1 + L2, 0.0])),
        ("basic_folded_L", np.array([0.0, math.pi / 2.0, -math.pi / 2.0]), np.array([L2, 0.0, L1])),
    ]


def check_validation_cases(geom: P2Geometry | None = None, atol: float = 1e-9) -> list[dict[str, Any]]:
    """Run FK validation cases and return result dictionaries."""

    geom = geom or load_geometry()
    rows: list[dict[str, Any]] = []
    for name, q, expected in validation_cases(geom):
        actual = fk_p2(q, geom)
        error_m = float(np.linalg.norm(actual - expected))
        rows.append(
            {
                "case": name,
                "q_rad": q.tolist(),
                "actual_m": actual.tolist(),
                "expected_m": expected.tolist(),
                "error_m": error_m,
                "passed": bool(error_m <= atol),
            }
        )
    return rows
