#!/usr/bin/env python3
"""Validate P1 rigid FK convention."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p1_fk import L1_EFFECTIVE_M, validate_p1_fk


def main() -> int:
    print(f"P1 FK check: L1_effective_m={L1_EFFECTIVE_M:.6f}")
    results = validate_p1_fk()
    failed = False
    for result in results:
        actual = np.asarray(result["actual_m"], dtype=float)
        expected = np.asarray(result["expected_m"], dtype=float)
        status = "PASS" if result["pass"] else "FAIL"
        print(
            f"{status} {result['name']}: "
            f"q=[{result['q1_rad']:.9f}, {result['q2_rad']:.9f}] "
            f"actual={actual.tolist()} expected={expected.tolist()} "
            f"error_m={result['error_m']:.3e}"
        )
        failed = failed or not bool(result["pass"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
