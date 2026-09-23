from __future__ import annotations

"""Calibration-gated builder scaffold for clean_dataset_1link_v2.

This script intentionally does not infer calibration from the current v1 data.
The active robust-2mm goal found that v1 is not enough for an absolute 2 mm
claim, so v2 may only be created from processed rows plus an explicit
calibration specification that either says the correction has already been
applied to the rows being packaged or provides a reviewed transform for this
builder to apply.

Safe default:
- `--write-template PATH` writes a calibration spec template and exits.
- `--calibration-spec PATH` validates the spec.
- The script refuses to create v2 unless the spec states either
  `dataset_transform_status == "already_applied_to_rows"` or
  `dataset_transform_status == "builder_applies_transform_to_rows"`.

No model is trained and no raw/source files are joined into train/eval
features.
"""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v2"
REQUIRED_LABEL_COLUMNS = ["p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"]
FEATURE_LEAKAGE_PREFIXES = ("p_tracker", "label_tracker", "p_vive")
ACCEPTED_TRANSFORM_STATUSES = {"already_applied_to_rows", "builder_applies_transform_to_rows"}
TRANSFORM_COLUMN_TRIPLES = [
    ("p_target_x_mm", "p_target_y_mm", "p_target_z_mm"),
    ("p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"),
    ("p_tracker_vertical_home_x_mm", "p_tracker_vertical_home_y_mm", "p_tracker_vertical_home_z_mm"),
]


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def spec_template() -> dict[str, Any]:
    return {
        "schema_version": "clean_dataset_1link_v2_calibration_spec_v1",
        "dataset_transform_status": "TODO: already_applied_to_rows or builder_applies_transform_to_rows",
        "calibration_method": "TODO: e.g. rigid tracker-to-robot/base transform from measured calibration points",
        "calibration_date": "TODO: YYYY-MM-DD",
        "calibration_source": "TODO: measurement file, runbook, or notebook path",
        "tracker_to_base_transform_version": "TODO: version string",
        "robot_base_transform_version": "TODO: version string",
        "robot_base_calibrated": True,
        "tracker_to_base_calibrated": True,
        "absolute_error_valid_by_source_group": {
            "test-260523-2": True,
            "test-260524-1": True,
        },
        "calibration_method_by_source_group": {
            "test-260523-2": "TODO",
            "test-260524-1": "TODO",
        },
        "target_provenance_policy": "TODO: unified or normalized target provenance",
        "transform_application_note": "TODO: describe exactly how p_target and p_vive rows were transformed before packaging v2",
        "tracker_to_base_transform_matrix": "TODO: required when dataset_transform_status is builder_applies_transform_to_rows",
        "reviewer": "TODO",
    }


def validate_transform_matrix(matrix: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(matrix, list) or len(matrix) != 4:
        return ["tracker_to_base_transform_matrix must be a 4x4 numeric list"]
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 4:
            errors.append(f"tracker_to_base_transform_matrix row {row_index} must have 4 values")
            continue
        for column_index, value in enumerate(row):
            if not isinstance(value, (int, float)):
                errors.append(f"tracker_to_base_transform_matrix[{row_index}][{column_index}] must be numeric")
    if errors:
        return errors
    bottom = matrix[3]
    if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(bottom, [0.0, 0.0, 0.0, 1.0])):
        errors.append("tracker_to_base_transform_matrix bottom row must be [0, 0, 0, 1]")
    return errors


def validate_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_text = [
        "calibration_method",
        "calibration_date",
        "calibration_source",
        "tracker_to_base_transform_version",
        "robot_base_transform_version",
        "target_provenance_policy",
        "transform_application_note",
    ]
    transform_status = spec.get("dataset_transform_status")
    if transform_status not in ACCEPTED_TRANSFORM_STATUSES:
        errors.append(
            "dataset_transform_status must be one of "
            + ", ".join(sorted(ACCEPTED_TRANSFORM_STATUSES))
        )
    for key in required_text:
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip() or value.strip().startswith("TODO"):
            errors.append(f"{key} must be a non-TODO string")
    if spec.get("robot_base_calibrated") is not True:
        errors.append("robot_base_calibrated must be true")
    if spec.get("tracker_to_base_calibrated") is not True:
        errors.append("tracker_to_base_calibrated must be true")
    absolute_valid = spec.get("absolute_error_valid_by_source_group")
    if not isinstance(absolute_valid, dict) or not absolute_valid or not all(value is True for value in absolute_valid.values()):
        errors.append("absolute_error_valid_by_source_group must be a non-empty all-true mapping")
    if transform_status == "builder_applies_transform_to_rows":
        errors.extend(validate_transform_matrix(spec.get("tracker_to_base_transform_matrix")))
    return errors


def validate_input_dataset(input_root: Path) -> dict[str, Any]:
    required = [
        input_root / "dataset.csv",
        input_root / "feature_columns.txt",
        input_root / "label_columns.txt",
        input_root / "manifest.json",
        input_root / "session_summary.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input dataset files: " + ", ".join(str(path) for path in missing))

    feature_columns = _read_lines(input_root / "feature_columns.txt")
    label_columns = _read_lines(input_root / "label_columns.txt")
    leakage = [column for column in feature_columns if any(column.startswith(prefix) for prefix in FEATURE_LEAKAGE_PREFIXES)]
    if leakage:
        raise ValueError(f"Feature columns contain label/tracker leakage candidates: {leakage}")
    if label_columns != REQUIRED_LABEL_COLUMNS:
        raise ValueError(f"Unexpected label columns: {label_columns}")
    return {
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "manifest": _load_json(input_root / "manifest.json"),
        "rows": _read_csv(input_root / "dataset.csv"),
        "session_summary_rows": _read_csv(input_root / "session_summary.csv"),
    }


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_template(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_float(value: Any, column: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric before applying calibration transform, got {value!r}") from exc


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _apply_transform_to_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def apply_builder_transform(rows: list[dict[str, Any]], matrix: list[list[float]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformed_rows: list[dict[str, Any]] = []
    transformed_columns: list[list[str]] = []
    for row_index, row in enumerate(rows):
        out = dict(row)
        for columns in TRANSFORM_COLUMN_TRIPLES:
            if not all(column in out and str(out[column]).strip() for column in columns):
                continue
            point = tuple(_to_float(out[column], column) for column in columns)
            transformed = _apply_transform_to_point(matrix, point)  # type: ignore[arg-type]
            for column, value in zip(columns, transformed):
                out[column] = _format_float(value)
            if row_index == 0:
                transformed_columns.append(list(columns))
        transformed_rows.append(out)
    return transformed_rows, {
        "rows_transformed": len(transformed_rows),
        "column_triples": transformed_columns,
    }


def build_v2(input_root: Path, output_root: Path, calibration_spec_path: Path) -> dict[str, Any]:
    spec = _load_json(calibration_spec_path)
    errors = validate_spec(spec)
    if errors:
        raise ValueError("Calibration spec is not acceptable for v2:\n- " + "\n- ".join(errors))

    data = validate_input_dataset(input_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")

    transform_status = spec["dataset_transform_status"]
    rows = data["rows"]
    transform_application: dict[str, Any] = {
        "dataset_transform_status": transform_status,
        "applied_by_builder": False,
    }
    if transform_status == "builder_applies_transform_to_rows":
        matrix = [[float(value) for value in row] for row in spec["tracker_to_base_transform_matrix"]]
        rows, transform_application = apply_builder_transform(rows, matrix)
        transform_application["applied_by_builder"] = True

    output_root.mkdir(parents=True)
    dataset_fieldnames = list(rows[0].keys()) if rows else []
    session_fieldnames = list(data["session_summary_rows"][0].keys()) if data["session_summary_rows"] else []
    _write_csv(output_root / "dataset.csv", rows, dataset_fieldnames)
    _write_csv(output_root / "session_summary.csv", data["session_summary_rows"], session_fieldnames)
    _write_lines(output_root / "feature_columns.txt", data["feature_columns"])
    _write_lines(output_root / "label_columns.txt", data["label_columns"])
    shutil.copy2(calibration_spec_path, output_root / "calibration_spec.json")

    manifest = dict(data["manifest"])
    manifest.update(
        {
            "schema_version": "clean_dataset_1link_v2_calibration_gated",
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source_dataset_root": _rel(input_root),
            "output_root": _rel(output_root),
            "dataset_csv": _rel(output_root / "dataset.csv"),
            "build_script": _rel(Path(__file__).resolve()),
            "calibration_spec": _rel(output_root / "calibration_spec.json"),
            "transform_application": transform_application,
        }
    )
    coordinate_policy = dict(manifest.get("coordinate_policy", {}))
    coordinate_policy.update(
        {
            "robot_base_calibrated": True,
            "tracker_to_base_calibrated": True,
            "absolute_error_valid_by_source_group": spec["absolute_error_valid_by_source_group"],
            "calibration_method": spec["calibration_method"],
            "calibration_date": spec["calibration_date"],
            "calibration_source": spec["calibration_source"],
            "tracker_to_base_transform_version": spec["tracker_to_base_transform_version"],
            "robot_base_transform_version": spec["robot_base_transform_version"],
            "calibration_method_by_source_group": spec.get("calibration_method_by_source_group", {}),
            "target_provenance_policy": spec["target_provenance_policy"],
            "transform_application_note": spec["transform_application_note"],
            "dataset_transform_status": transform_status,
            "builder_applied_transform": transform_application["applied_by_builder"],
        }
    )
    if "tracker_to_base_transform_matrix" in spec:
        coordinate_policy["tracker_to_base_transform_matrix"] = spec["tracker_to_base_transform_matrix"]
    manifest["coordinate_policy"] = coordinate_policy
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = "\n".join(
        [
            "# clean_dataset_1link_v2",
            "",
            "This dataset was packaged by `build_clean_dataset_1link_v2.py` after a calibration spec declared or provided the required coordinate transform.",
            "",
            "This builder does not infer calibration and does not train a model.",
            "",
            f"- source dataset: `{_rel(input_root)}`",
            f"- calibration spec: `{_rel(output_root / 'calibration_spec.json')}`",
            f"- transform status: `{transform_status}`",
            f"- builder applied transform: `{transform_application['applied_by_builder']}`",
            f"- rows: `{len(rows)}`",
            "",
        ]
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output_root": _rel(output_root),
        "rows": len(rows),
        "calibration_spec": _rel(output_root / "calibration_spec.json"),
        "transform_application": transform_application,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--calibration-spec")
    parser.add_argument("--write-template")
    args = parser.parse_args()

    if args.write_template:
        write_template(Path(args.write_template))
        print(json.dumps({"template": args.write_template}, indent=2))
        return 0

    if not args.calibration_spec:
        raise SystemExit("Refusing to build v2 without --calibration-spec. Use --write-template first.")

    result = build_v2(Path(args.input_root), Path(args.output_root), Path(args.calibration_spec))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
