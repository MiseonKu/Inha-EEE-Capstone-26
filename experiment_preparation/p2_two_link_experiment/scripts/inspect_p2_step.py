"""Optional STEP inspection for the P2 two-link assembly."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEP_PATH = ROOT / "source_step" / "2_LINK_SO101_ASSEMBLY.step"
OUTPUT = ROOT / "outputs" / "pre_imu_p2" / "step_inspection.json"


def try_freecad() -> dict | None:
    try:
        import Part  # type: ignore
    except Exception:
        return None
    shape = Part.Shape()
    shape.read(str(STEP_PATH))
    bbox = shape.BoundBox
    return {
        "tool": "FreeCAD Part",
        "bbox": {"x": bbox.XLength, "y": bbox.YLength, "z": bbox.ZLength},
        "note": "CAD units are file-dependent; this is not FK authority.",
    }


def try_cadquery() -> dict | None:
    try:
        import cadquery as cq  # type: ignore
    except Exception:
        return None
    shape = cq.importers.importStep(str(STEP_PATH))
    bbox = shape.val().BoundingBox()
    return {
        "tool": "cadquery",
        "bbox": {"x": bbox.xlen, "y": bbox.ylen, "z": bbox.zlen},
        "note": "CAD units are file-dependent; this is not FK authority.",
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "step_file": STEP_PATH.relative_to(ROOT).as_posix(),
        "step_exists": STEP_PATH.exists(),
        "step_measurement_succeeded": False,
        "mesh_conversion_succeeded": False,
        "warning": "",
        "authoritative_geometry_note": "Do not infer FK length, axes, yaw sign, or pitch sign from STEP.",
    }
    if not STEP_PATH.exists():
        result["warning"] = "STEP file missing."
    else:
        measured = try_freecad() or try_cadquery()
        if measured is None:
            result["warning"] = "FreeCAD/CadQuery unavailable; STEP dimensions were not measured."
        else:
            result.update(measured)
            result["step_measurement_succeeded"] = True
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
