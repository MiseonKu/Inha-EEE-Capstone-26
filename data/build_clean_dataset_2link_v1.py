from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "source_processed" / "2link_synthetic"
TARGET_PLAN_PATH = SOURCE_ROOT / "synthetic_only_target_plan_5000.csv"
SUPERVISED_PATH = SOURCE_ROOT / "synthetic_only_supervised_samples_5000.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_2link_v1"

FEATURE_COLUMNS = [
    "q1_cmd_deg",
    "q2_cmd_deg",
    "q3_cmd_deg",
    "q1_encoder_deg",
    "q2_encoder_deg",
    "q3_encoder_deg",
    "p_target_x_mm",
    "p_target_y_mm",
    "p_target_z_mm",
]

LABEL_COLUMNS = [
    "p_vive_x_mm",
    "p_vive_y_mm",
    "p_vive_z_mm",
]

METADATA_COLUMNS = [
    "source_group",
    "source_session",
    "session_id",
    "target_id",
    "sample_id",
    "planned_order_index",
    "timestamp_utc",
    "settle_time_s",
    "settle_start_s",
    "settle_end_s",
    "sample_count",
    "kinematic_model_version",
]

RAW_LABEL_COLUMNS = [
    "p_tracker_relative_x_mm",
    "p_tracker_relative_y_mm",
    "p_tracker_relative_z_mm",
]

FRAME_DIAGNOSTIC_COLUMNS = [
    "target_label_residual_x_mm",
    "target_label_residual_y_mm",
    "target_label_residual_z_mm",
    "target_label_residual_norm_mm",
]

QUALITY_COLUMNS = [
    "quality_flags",
    "tracker_valid_ratio",
    "label_window_std_mm",
    "label_window_max_span_mm",
    "label_sample_count",
    "synthetic_label",
]

OUTPUT_COLUMNS = [
    *METADATA_COLUMNS,
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
    *RAW_LABEL_COLUMNS,
    *FRAME_DIAGNOSTIC_COLUMNS,
    *QUALITY_COLUMNS,
]

EXPECTED_QUALITY_FLAG = "synthetic_nonlinear_interp"
MIN_TRACKER_VALID_RATIO = 0.99
FRAME_ALIGNMENT_NAME = "kabsch_rigid_label_to_target_frame_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csv_ready_row(row, fieldnames) for row in rows)


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _csv_ready_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _csv_ready_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {key: _csv_ready_cell(row.get(key)) for key in fieldnames}


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _float_or_none(value: str | None) -> float | None:
    text = str(value).strip() if value is not None else ""
    if not text or text.upper() == "NA":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: str | None) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _target_plan_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_id = (row.get("target_id") or "").strip()
        if not target_id.startswith("SYNRAW"):
            continue

        required = {
            "planned_order_index": _int_or_none(row.get("planned_order_index")),
            "q1_cmd_deg": _float_or_none(row.get("q1_commanded_delta_deg")),
            "q2_cmd_deg": _float_or_none(row.get("q2_commanded_delta_deg")),
            "q3_cmd_deg": _float_or_none(row.get("q3_commanded_delta_deg")),
            "p_target_x_mm": _float_or_none(row.get("target_x_mm")),
            "p_target_y_mm": _float_or_none(row.get("target_y_mm")),
            "p_target_z_mm": _float_or_none(row.get("target_z_mm")),
        }
        if all(value is not None for value in required.values()):
            out[target_id] = required
    return out


def _row_reject_reason(row: dict[str, str], target_features: dict[str, dict[str, Any]]) -> str | None:
    target_id = (row.get("target_id") or "").strip()
    if not target_id.startswith("SYNRAW"):
        return "target_id_not_SYNRAW"
    if target_id not in target_features:
        return "missing_target_features"
    if (row.get("quality_flags") or "").strip() != EXPECTED_QUALITY_FLAG:
        return "quality_flags_not_synthetic_nonlinear_interp"

    valid_ratio = _float_or_none(row.get("tracker_valid_ratio"))
    if valid_ratio is None or valid_ratio < MIN_TRACKER_VALID_RATIO:
        return "tracker_valid_ratio_below_threshold"

    sample_count = _int_or_none(row.get("label_sample_count"))
    if sample_count is None or sample_count <= 0:
        return "label_sample_count_not_positive"

    required_supervised = [
        "feature_pan_logical_deg",
        "feature_lift_logical_deg",
        "feature_elbow_logical_deg",
        "label_window_mean_x_mm_relative",
        "label_window_mean_y_mm_relative",
        "label_window_mean_z_mm_relative",
        "label_window_std_mm",
        "label_window_max_span_mm",
    ]
    if any(_float_or_none(row.get(name)) is None for name in required_supervised):
        return "missing_required_supervised_numeric"

    return None


def _kabsch_label_to_target(
    raw_label_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_centroid = raw_label_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = raw_label_points - source_centroid
    target_centered = target_points - target_centroid

    h = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    reflection_corrected = False
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
        reflection_corrected = True

    translation = target_centroid - source_centroid @ rotation.T
    aligned = raw_label_points @ rotation.T + translation
    residual = aligned - target_points
    axis_rmse = np.sqrt(np.mean(residual * residual, axis=0))
    residual_norms = np.linalg.norm(residual, axis=1)

    diagnostics = {
        "fit_rows": int(len(raw_label_points)),
        "rotation_matrix": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "source_centroid_mm": source_centroid.tolist(),
        "target_centroid_mm": target_centroid.tolist(),
        "determinant": float(np.linalg.det(rotation)),
        "reflection_corrected": reflection_corrected,
        "singular_values": singular_values.tolist(),
        "rmse_3d_mm": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "axis_rmse_mm": {
            "x": float(axis_rmse[0]),
            "y": float(axis_rmse[1]),
            "z": float(axis_rmse[2]),
        },
        "residual_norm_mm": {
            "min": float(np.min(residual_norms)),
            "mean": float(np.mean(residual_norms)),
            "median": float(np.median(residual_norms)),
            "max": float(np.max(residual_norms)),
        },
    }
    return aligned, residual, residual_norms, diagnostics


def _build_candidate_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_rows = _read_csv(TARGET_PLAN_PATH)
    supervised_rows = _read_csv(SUPERVISED_PATH)
    target_features = _target_plan_by_id(target_rows)

    rejected: dict[str, list[str]] = {}
    candidates: list[dict[str, Any]] = []
    raw_label_points: list[list[float]] = []
    target_points: list[list[float]] = []
    session_ids = sorted({(row.get("session_id") or "").strip() for row in supervised_rows if row.get("session_id")})
    source_session = session_ids[0] if len(session_ids) == 1 else SOURCE_ROOT.name

    for row in supervised_rows:
        target_id = (row.get("target_id") or "").strip()
        reject_reason = _row_reject_reason(row, target_features)
        if reject_reason is not None:
            rejected.setdefault(reject_reason, []).append(target_id)
            continue

        target = target_features[target_id]
        raw_x = float(_float_or_none(row.get("label_window_mean_x_mm_relative")))
        raw_y = float(_float_or_none(row.get("label_window_mean_y_mm_relative")))
        raw_z = float(_float_or_none(row.get("label_window_mean_z_mm_relative")))

        candidate = {
            "source_group": SOURCE_ROOT.name,
            "source_session": source_session,
            "session_id": row.get("session_id") or source_session,
            "target_id": target_id,
            "sample_id": _int_or_none(row.get("sample_id")),
            "planned_order_index": target["planned_order_index"],
            "timestamp_utc": row.get("timestamp_utc", ""),
            "settle_time_s": _float_or_none(row.get("t_session_s")),
            "settle_start_s": _float_or_none(row.get("label_window_start_s")),
            "settle_end_s": _float_or_none(row.get("label_window_end_s")),
            "sample_count": _int_or_none(row.get("label_sample_count")),
            "kinematic_model_version": row.get("feature_kinematic_model_version", ""),
            "q1_cmd_deg": target["q1_cmd_deg"],
            "q2_cmd_deg": target["q2_cmd_deg"],
            "q3_cmd_deg": target["q3_cmd_deg"],
            "q1_encoder_deg": _float_or_none(row.get("feature_pan_logical_deg")),
            "q2_encoder_deg": _float_or_none(row.get("feature_lift_logical_deg")),
            "q3_encoder_deg": _float_or_none(row.get("feature_elbow_logical_deg")),
            "p_target_x_mm": target["p_target_x_mm"],
            "p_target_y_mm": target["p_target_y_mm"],
            "p_target_z_mm": target["p_target_z_mm"],
            "p_tracker_relative_x_mm": raw_x,
            "p_tracker_relative_y_mm": raw_y,
            "p_tracker_relative_z_mm": raw_z,
            "quality_flags": row.get("quality_flags", ""),
            "tracker_valid_ratio": _float_or_none(row.get("tracker_valid_ratio")),
            "label_window_std_mm": _float_or_none(row.get("label_window_std_mm")),
            "label_window_max_span_mm": _float_or_none(row.get("label_window_max_span_mm")),
            "label_sample_count": _int_or_none(row.get("label_sample_count")),
            "synthetic_label": True,
        }
        candidates.append(candidate)
        raw_label_points.append([raw_x, raw_y, raw_z])
        target_points.append([target["p_target_x_mm"], target["p_target_y_mm"], target["p_target_z_mm"]])

    candidates.sort(key=lambda row: (int(row.get("planned_order_index") or 10**12), str(row.get("target_id", ""))))
    raw_arr = np.asarray([[row["p_tracker_relative_x_mm"], row["p_tracker_relative_y_mm"], row["p_tracker_relative_z_mm"]] for row in candidates], dtype=float)
    target_arr = np.asarray([[row["p_target_x_mm"], row["p_target_y_mm"], row["p_target_z_mm"]] for row in candidates], dtype=float)
    aligned, residual, residual_norms, frame_diagnostics = _kabsch_label_to_target(raw_arr, target_arr)

    for index, row in enumerate(candidates):
        row["p_vive_x_mm"] = float(aligned[index, 0])
        row["p_vive_y_mm"] = float(aligned[index, 1])
        row["p_vive_z_mm"] = float(aligned[index, 2])
        row["target_label_residual_x_mm"] = float(residual[index, 0])
        row["target_label_residual_y_mm"] = float(residual[index, 1])
        row["target_label_residual_z_mm"] = float(residual[index, 2])
        row["target_label_residual_norm_mm"] = float(residual_norms[index])

    summary = {
        "source_group": SOURCE_ROOT.name,
        "source_root": _rel(SOURCE_ROOT),
        "source_session": source_session,
        "target_plan_csv": _rel(TARGET_PLAN_PATH),
        "supervised_samples_csv": _rel(SUPERVISED_PATH),
        "target_plan_rows_total": len(target_rows),
        "supervised_rows_total": len(supervised_rows),
        "target_feature_SYNRAW_rows": len(target_features),
        "clean_rows_added": len(candidates),
        "rejected_rows_total": sum(len(values) for values in rejected.values()),
        "rejected_by_reason": {key: len(values) for key, values in sorted(rejected.items())},
        "excluded_target_ids_by_reason": {
            key: sorted(value for value in values if value)
            for key, values in sorted(rejected.items())
        },
        "quality_flags_required": EXPECTED_QUALITY_FLAG,
        "min_tracker_valid_ratio": MIN_TRACKER_VALID_RATIO,
        "synthetic_label": True,
        "frame_alignment": frame_diagnostics,
    }
    return candidates, summary


def _source_line(summary: dict[str, Any]) -> str:
    return (
        f"- `{summary['source_root']}`: target plan {summary['target_plan_rows_total']} rows, "
        f"supervised {summary['supervised_rows_total']} rows, clean {summary['clean_rows_added']} rows, "
        f"rejected {summary['rejected_rows_total']} rows"
    )


def _write_readme(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    frame = summary["frame_alignment"]
    readme = f"""# clean_dataset_2link_v1

`clean_dataset_2link_v1`는 `data/source_processed/2link_synthetic`의 2link synthetic source-processed data를 사용해서 만든 ML 학습용 데이터셋이다.
`clean_dataset_1link_v1`과 섞지 않고 2link/3축 전용 schema로 분리했다.
한 row는 하나의 synthetic settled target 결과를 의미한다.

## 전처리/정제 방식

사용한 source:

{_source_line(summary)}

사용한 파일:

- `{summary['target_plan_csv']}`
- `{summary['supervised_samples_csv']}`

처리 순서:

1. `synthetic_only_target_plan_5000.csv`에서 `SYNRAW*` target command/position 정보를 읽는다.
2. `synthetic_only_supervised_samples_5000.csv`에서 같은 `target_id`의 synthetic settled label row를 읽는다.
3. 학습에 사용할 수 있는 row만 남긴다.
4. q1/q2/q3 명령각, settled encoder/logical angle, target 위치를 feature로 저장한다.
5. raw synthetic label 좌표에는 전역 rigid Kabsch 정합을 적용해서 target/model-FK frame 쪽으로 맞춘다.
6. 최종 `dataset.csv`, `feature_columns.txt`, `label_columns.txt`, `manifest.json`, `session_summary.csv`를 저장한다.

row filter:

- `target_id`가 `SYNRAW`로 시작
- `quality_flags == "{EXPECTED_QUALITY_FLAG}"`
- `tracker_valid_ratio >= {MIN_TRACKER_VALID_RATIO}`
- `label_sample_count > 0`
- target command/position, encoder angle, raw label 좌표가 모두 numeric

정제 결과:

- 최종 학습 row: {len(rows)}개
- rejected row: {summary['rejected_rows_total']}개

## 학습 문제 정의

모델은 에러를 직접 학습하지 않는다. 모델은 frame 정합된 synthetic Tracker 위치를 학습한다.

입력 `X`:

- `q1_cmd_deg`
- `q2_cmd_deg`
- `q3_cmd_deg`
- `q1_encoder_deg`
- `q2_encoder_deg`
- `q3_encoder_deg`
- `p_target_x_mm`
- `p_target_y_mm`
- `p_target_z_mm`

출력 `y`:

- `p_vive_x_mm`
- `p_vive_y_mm`
- `p_vive_z_mm`

즉 모델이 학습하는 관계는 아래와 같다.

```text
2link/3축 명령각 + settled 엔코더각 + 의도한 target 위치
  -> frame 정합된 synthetic Tracker settled 위치
```

에러 보정은 학습 이후 단계에서 수행한다.

```text
predicted_actual_position = model(X)
error = predicted_actual_position - target_position
```

## 좌표 정책

원본 synthetic label은 `label_window_mean_*_mm_relative`이다. 이 raw label 좌표는
`p_tracker_relative_*_mm` 컬럼에 보존했다.

최종 label `p_vive_*_mm`에는 global rigid Kabsch 정합을 적용했다.

- frame alignment name: `{FRAME_ALIGNMENT_NAME}`
- fit rows: {frame['fit_rows']}
- 3D RMSE after alignment: {frame['rmse_3d_mm']:.3f} mm
- axis RMSE: x={frame['axis_rmse_mm']['x']:.3f} mm, y={frame['axis_rmse_mm']['y']:.3f} mm, z={frame['axis_rmse_mm']['z']:.3f} mm

주의:

- 이 데이터셋은 `clean_dataset_1link_v1`의 exact 90도 축 매핑을 그대로 사용하지 않는다.
- v2는 synthetic-only 2link data이므로 `synthetic_label = true`로 기록한다.
- Kabsch 정합은 frame registration이며, nonlinear synthetic residual은 label에 남아 있다.
- 원본 raw label 좌표는 `p_tracker_relative_*_mm`, 정합 후 target 대비 residual은 `target_label_residual_*_mm`에 보존했다.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def _write_manifest(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    manifest = {
        "schema_version": "clean_dataset_2link_v1_kabsch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": _rel(SOURCE_ROOT),
        "source_roots": [_rel(SOURCE_ROOT)],
        "output_root": _rel(OUTPUT_ROOT),
        "dataset_csv": _rel(OUTPUT_ROOT / "dataset.csv"),
        "build_script": _rel(Path(__file__).resolve()),
        "row_policy": "one row per synthetic settled target from package-local source_processed/2link_synthetic",
        "row_filter": {
            "target_id_prefix": "SYNRAW",
            "quality_flags": EXPECTED_QUALITY_FLAG,
            "tracker_valid_ratio_min": MIN_TRACKER_VALID_RATIO,
            "label_sample_count_positive": True,
            "requires_target_features": True,
            "requires_encoder_features": True,
            "requires_raw_label": True,
        },
        "coordinate_policy": {
            "position_unit": "mm",
            "target_source": "synthetic_only_target_plan_5000.csv target_x_mm,target_y_mm,target_z_mm",
            "raw_tracker_label_source": "synthetic_only_supervised_samples_5000.csv label_window_mean_*_mm_relative",
            "final_label_columns": LABEL_COLUMNS,
            "label_coordinate_frame": "kabsch_aligned_2link_target_model_fk_frame",
            "frame_alignment_name": FRAME_ALIGNMENT_NAME,
            "frame_alignment_fit_scope": "all clean rows in data/source_processed/2link_synthetic",
            "frame_alignment": summary["frame_alignment"],
            "centroid_shift_applied": True,
            "raw_label_preserved": True,
            "raw_label_columns": RAW_LABEL_COLUMNS,
            "synthetic_label": True,
            "robot_base_calibrated": False,
            "tracker_to_base_calibrated": False,
            "calibration_status": "SYNTHETIC_FRAME_REGISTERED",
        },
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "raw_label_columns": RAW_LABEL_COLUMNS,
        "frame_diagnostic_columns": FRAME_DIAGNOSTIC_COLUMNS,
        "quality_columns": QUALITY_COLUMNS,
        "total_clean_rows": len(rows),
        "total_supervised_rows": summary["supervised_rows_total"],
        "total_rejected_rows": summary["rejected_rows_total"],
        "session_summaries": [summary],
        "source_counts_by_source_group": {
            summary["source_group"]: {
                "target_plan_rows": summary["target_plan_rows_total"],
                "supervised_rows": summary["supervised_rows_total"],
                "target_feature_SYNRAW_rows": summary["target_feature_SYNRAW_rows"],
                "clean_rows": summary["clean_rows_added"],
                "rejected_rows": summary["rejected_rows_total"],
            }
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global SOURCE_ROOT, TARGET_PLAN_PATH, SUPERVISED_PATH, OUTPUT_ROOT

    parser = argparse.ArgumentParser(description="Build clean_dataset_2link_v1 from package-local source_processed inputs.")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    SOURCE_ROOT = Path(args.source_root)
    TARGET_PLAN_PATH = SOURCE_ROOT / "synthetic_only_target_plan_5000.csv"
    SUPERVISED_PATH = SOURCE_ROOT / "synthetic_only_supervised_samples_5000.csv"
    OUTPUT_ROOT = Path(args.output_root)
    if OUTPUT_ROOT.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output root: {OUTPUT_ROOT}. Pass --overwrite or choose --output-root.")
    if OUTPUT_ROOT.exists():
        import shutil

        shutil.rmtree(OUTPUT_ROOT)

    rows, summary = _build_candidate_rows()
    _write_csv(OUTPUT_ROOT / "dataset.csv", rows, OUTPUT_COLUMNS)
    _write_csv(OUTPUT_ROOT / "session_summary.csv", [summary], list(summary.keys()))
    _write_lines(OUTPUT_ROOT / "feature_columns.txt", FEATURE_COLUMNS)
    _write_lines(OUTPUT_ROOT / "label_columns.txt", LABEL_COLUMNS)
    _write_readme(rows, summary)
    _write_manifest(rows, summary)
    print(f"wrote {_rel(OUTPUT_ROOT / 'dataset.csv')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
