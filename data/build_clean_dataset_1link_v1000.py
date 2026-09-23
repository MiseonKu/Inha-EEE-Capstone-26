from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v1000"
DEFAULT_SOURCE_GROUP = "test-260524-1"
SCHEMA_VERSION = "clean_dataset_1link_v1000_single_session_v1"


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def session_summary(rows: list[dict[str, str]], source_group: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        session = row.get("source_session") or row.get("session_id") or "unknown"
        counts[session] = counts.get(session, 0) + 1
    return [
        {
            "source_group": source_group,
            "source_session": session,
            "clean_rows_added": count,
            "supervised_rows_total": count,
            "rejected_rows_total": 0,
            "target_source": "clean_dataset_1link_v2 dataset.csv",
        }
        for session, count in sorted(counts.items())
    ]


def write_readme(output_root: Path, input_root: Path, source_group: str, row_count: int) -> None:
    text = f"""# clean_dataset_1link_v1000

This dataset is the submission-package copy of the final 1-link v1000 dataset.
It is built by extracting source group `{source_group}` from the included
`clean_dataset_1link_v2` processed dataset.

## Row Policy

- source dataset: `{rel(input_root)}`
- source group: `{source_group}`
- final clean rows: `{row_count}`
- one row represents one settled target sample

## Coordinate Policy

The labels are Vive Tracker settled positions in millimeters after the fixed
1-link axis mapping used by the final experiments:

```text
p_vive_x_mm =  tracker_y
p_vive_y_mm = -tracker_x
p_vive_z_mm =  tracker_z
```

This v1000 subset does not introduce a new fitted calibration transform. It
keeps the processed label frame documented in the parent 1-link v2 dataset and
records the subset extraction for reproducibility.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def build_dataset(input_root: Path, output_root: Path, source_group: str, overwrite: bool) -> dict[str, Any]:
    dataset_path = input_root / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {dataset_path}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
        shutil.rmtree(output_root)

    fieldnames, source_rows = read_csv(dataset_path)
    rows = [row for row in source_rows if row.get("source_group") == source_group]
    if not rows:
        raise ValueError(f"No rows found for source_group={source_group!r} in {dataset_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "dataset.csv", rows, fieldnames)
    summaries = session_summary(rows, source_group)
    write_csv(output_root / "session_summary.csv", summaries, list(summaries[0].keys()))
    feature_columns = read_lines(input_root / "feature_columns.txt")
    label_columns = read_lines(input_root / "label_columns.txt")
    write_lines(output_root / "feature_columns.txt", feature_columns)
    write_lines(output_root / "label_columns.txt", label_columns)
    write_readme(output_root, input_root, source_group, len(rows))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_dataset_root": rel(input_root),
        "source_group": source_group,
        "output_root": rel(output_root),
        "dataset_csv": rel(output_root / "dataset.csv"),
        "build_script": rel(Path(__file__).resolve()),
        "row_policy": "package-local extraction of one source_group from clean_dataset_1link_v2",
        "coordinate_policy": {
            "position_unit": "mm",
            "label_coordinate_frame": "test-260524-1_vertical_home_relative_tracker_frame_axis_mapped",
            "axis_mapping_name": "origin_locked_exact_90_candidate_A",
            "axis_mapping_formula": {
                "p_vive_x_mm": "label_tracker_y_mm_relative_vertical_home",
                "p_vive_y_mm": "-label_tracker_x_mm_relative_vertical_home",
                "p_vive_z_mm": "label_tracker_z_mm_relative_vertical_home",
            },
        },
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "total_clean_rows": len(rows),
        "session_summaries": summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_root": rel(output_root), "rows": len(rows), "source_group": source_group}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build package-local 1-link v1000 dataset from included 1-link v2 data.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-group", default=DEFAULT_SOURCE_GROUP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.input_root, args.output_root, args.source_group, args.overwrite), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
