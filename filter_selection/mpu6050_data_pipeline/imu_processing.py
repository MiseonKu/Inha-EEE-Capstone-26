from __future__ import annotations

import csv
from pathlib import Path


ACCEL_RAW_FIELDS = ("ax_raw", "ay_raw", "az_raw")
GYRO_RAW_FIELDS = ("gx_raw", "gy_raw", "gz_raw")
TEMP_RAW_FIELDS = ("temp_raw", "temp_c")

ACCEL_UNIT_FIELDS = ("ax_mps2", "ay_mps2", "az_mps2")
GYRO_UNIT_FIELDS = ("gx_dps", "gy_dps", "gz_dps")
CSV_FIELDS = ("millis", *ACCEL_UNIT_FIELDS, *GYRO_UNIT_FIELDS, "temp_c")
PLOT_FIELDS = CSV_FIELDS[1:]

ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
GRAVITY_MPS2 = 9.80665
DEFAULT_EMA_ALPHA = 0.25


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def accel_raw_to_mps2(value: float) -> float:
    return (float(value) / ACCEL_LSB_PER_G) * GRAVITY_MPS2


def gyro_raw_to_dps(value: float) -> float:
    return float(value) / GYRO_LSB_PER_DPS


def _coerce_temp_c(row: dict[str, str]) -> float:
    temp_c_text = row.get("temp_c", "")
    if temp_c_text:
        return float(temp_c_text)

    temp_raw_text = row.get("temp_raw", "")
    if temp_raw_text:
        return (float(temp_raw_text) / 340.0) + 36.53

    raise KeyError("Row does not contain temp_c or temp_raw.")


def coerce_measurement_row(row: dict[str, str]) -> dict[str, float]:
    numeric = {"millis": float(row["millis"])}

    if all(field in row for field in ACCEL_UNIT_FIELDS + GYRO_UNIT_FIELDS):
        for field in ACCEL_UNIT_FIELDS + GYRO_UNIT_FIELDS:
            numeric[field] = float(row[field])
    elif all(field in row for field in ACCEL_RAW_FIELDS + GYRO_RAW_FIELDS):
        for raw_field, unit_field in zip(ACCEL_RAW_FIELDS, ACCEL_UNIT_FIELDS):
            numeric[unit_field] = accel_raw_to_mps2(float(row[raw_field]))
        for raw_field, unit_field in zip(GYRO_RAW_FIELDS, GYRO_UNIT_FIELDS):
            numeric[unit_field] = gyro_raw_to_dps(float(row[raw_field]))
    else:
        raise KeyError("Row does not contain recognized IMU fields.")

    numeric["temp_c"] = _coerce_temp_c(row)
    return numeric


def serial_row_to_measurement_row(row: dict[str, str]) -> dict[str, float]:
    return coerce_measurement_row(row)


def measurement_row_to_csv_row(row: dict[str, float]) -> dict[str, str]:
    csv_row = {"millis": str(int(round(row["millis"])))}

    for field in ACCEL_UNIT_FIELDS + GYRO_UNIT_FIELDS:
        csv_row[field] = f"{row[field]:.6f}"

    csv_row["temp_c"] = f"{row['temp_c']:.2f}"
    return csv_row


def raw_row_for_plot(row: dict[str, str]) -> dict[str, float]:
    return coerce_measurement_row(row)


class EmaSmoother:
    def __init__(self, fields: tuple[str, ...], alpha: float) -> None:
        self.fields = fields
        self.alpha = max(0.0, min(alpha, 1.0))
        self.state: dict[str, float] = {}

    def apply(self, row: dict[str, float]) -> dict[str, float]:
        if self.alpha >= 1.0:
            return dict(row)

        smoothed = dict(row)
        for field in self.fields:
            current = row[field]
            previous = self.state.get(field, current)
            value = (self.alpha * current) + ((1.0 - self.alpha) * previous)
            self.state[field] = value
            smoothed[field] = value

        return smoothed


def resolve_input_csv_path(csv_arg: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    candidate = Path(csv_arg)

    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    base_candidate = base_dir / candidate
    if base_candidate.exists():
        return base_candidate

    raw_candidate = base_dir.parent / "filter_results" / "experiment_data" / candidate
    if raw_candidate.exists():
        return raw_candidate

    return candidate
