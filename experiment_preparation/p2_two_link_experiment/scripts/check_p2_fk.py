"""Run required P2 rigid FK validation cases."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematics.p2_fk import check_validation_cases, load_geometry  # noqa: E402


def main() -> int:
    geom = load_geometry(ROOT / "config" / "p2_geometry.yaml")
    print("Authoritative P2 geometry")
    print(f"  L1_effective_m = {geom.L1_effective_m}")
    print(f"  L2_effective_m = {geom.L2_effective_m}")
    print(f"  link1_flexible_bar_length_m = {geom.link1_flexible_bar_length_m}")
    print(f"  link2_flexible_bar_length_m = {geom.link2_flexible_bar_length_m}")
    print(f"  link1_nonflex_offset_mm = {geom.link1_nonflex_offset_mm}")
    print(f"  link2_nonflex_offset_mm = {geom.link2_nonflex_offset_mm}")
    ok = True
    for row in check_validation_cases(geom):
        ok = ok and row["passed"]
        print(
            f"{row['case']}: actual={row['actual_m']} expected={row['expected_m']} "
            f"error_m={row['error_m']:.3e} {'PASS' if row['passed'] else 'FAIL'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
