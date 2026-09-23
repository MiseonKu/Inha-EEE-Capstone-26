from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from imu_pose_filters import (
    BasePoseFilter,
    PoseEstimate,
    VectorLowPassFilter,
    average_quaternions,
    build_pose_filter,
    relative_quaternion,
    pose_estimate_from_quaternion,
)
from imu_processing import ACCEL_UNIT_FIELDS, GYRO_UNIT_FIELDS, serial_row_to_measurement_row
from imu_save import (
    classify_header,
    list_ports,
    open_serial,
    parse_data_line,
    resolve_csv_path,
    send_command,
    wait_for_header,
)


AVAILABLE_FILTERS = ("none", "complementary", "kalman", "mahony", "madgwick")
SENSOR_FIELDS = (*ACCEL_UNIT_FIELDS, *GYRO_UNIT_FIELDS)
PLOT_AXES = ("roll_deg", "pitch_deg", "yaw_deg")
AXIS_LABELS = {
    "roll_deg": "Relative Roll (deg)",
    "pitch_deg": "Relative Pitch (deg)",
    "yaw_deg": "Relative Yaw (deg)",
}
ACCEL_COLORS = {
    ACCEL_UNIT_FIELDS[0]: "#d1495b",
    ACCEL_UNIT_FIELDS[1]: "#edae49",
    ACCEL_UNIT_FIELDS[2]: "#00798c",
}
GYRO_COLORS = {
    GYRO_UNIT_FIELDS[0]: "#30638e",
    GYRO_UNIT_FIELDS[1]: "#8e5572",
    GYRO_UNIT_FIELDS[2]: "#3b7a57",
}
FILTER_COLORS = {
    "none": "#7f5539",
    "complementary": "#d1495b",
    "kalman": "#00798c",
    "mahony": "#30638e",
    "madgwick": "#3b7a57",
}
ANGLE_LINE_STYLES = {
    "roll_deg": "-",
    "pitch_deg": "--",
    "yaw_deg": ":",
}
FILTER_COMMAND_ALIASES = {
    "n": "none",
    "none": "none",
    "c": "complementary",
    "comp": "complementary",
    "complementary": "complementary",
    "k": "kalman",
    "kalman": "kalman",
    "mh": "mahony",
    "mahony": "mahony",
    "mg": "madgwick",
    "madgwick": "madgwick",
    "all": "all",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MPU6050 calibration optionally, capture a baseline pose, and plot relative "
            "roll/pitch/yaw using selectable attitude filters."
        )
    )
    parser.add_argument("--port", help="Serial port such as COM14")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["none"],
        help=(
            "One or more filters to run: none complementary kalman mahony madgwick or all. "
            "Example: --filters none mahony"
        ),
    )
    parser.add_argument(
        "--calibrate-first",
        action="store_true",
        help="Send CALIBRATE to imu_calib.ino before baseline capture.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after opening the serial port.",
    )
    parser.add_argument(
        "--header-timeout",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for the IMU CSV header.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=float,
        default=40.0,
        help="Maximum seconds to wait for CALIBRATE to finish.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=5.0,
        help="Seconds to keep the IMU still in the desired baseline pose.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=20.0,
        help="Visible time window in seconds.",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=100,
        help="Plot refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--low-pass-alpha",
        type=float,
        default=0.6,
        help="Low-pass EMA alpha for accel/gyro preprocessing. Use 1.0 for no smoothing.",
    )
    parser.add_argument(
        "--disable-low-pass",
        action="store_true",
        help="Disable low-pass preprocessing regardless of --low-pass-alpha.",
    )
    parser.add_argument(
        "--csv",
        help=(
            "Optional processed-angle CSV output file name. Relative names are saved inside "
            "filter_results/experiment_data. If omitted, a timestamped CSV is auto-saved."
        ),
    )
    parser.add_argument(
        "--save-figure",
        help=(
            "Optional image file name for the relative-pose plot. Relative names are saved inside "
            "filter_results/experiment_data. If omitted, a timestamped PNG is auto-saved when the window closes."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable CSV and PNG output; only show the live relative-pose plot.",
    )
    parser.add_argument(
        "--complementary-alpha",
        type=float,
        default=0.85,
        help="Complementary filter gyro weight. Higher means more gyro trust.",
    )
    parser.add_argument(
        "--kalman-q-angle",
        type=float,
        default=0.001,
        help="Kalman process noise for angle state.",
    )
    parser.add_argument(
        "--kalman-q-bias",
        type=float,
        default=0.003,
        help="Kalman process noise for gyro bias state.",
    )
    parser.add_argument(
        "--kalman-r-measure",
        type=float,
        default=0.03,
        help="Kalman measurement noise for accel tilt measurements.",
    )
    parser.add_argument(
        "--mahony-kp",
        type=float,
        default=1.0,
        help="Mahony proportional gain.",
    )
    parser.add_argument(
        "--mahony-ki",
        type=float,
        default=0.0,
        help="Mahony integral gain.",
    )
    parser.add_argument(
        "--madgwick-beta",
        type=float,
        default=0.1,
        help="Madgwick beta gain.",
    )
    return parser.parse_args()


def expand_filter_names(filter_args: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in filter_args:
        normalized = value.strip().lower()
        names = AVAILABLE_FILTERS if normalized == "all" else (normalized,)
        for name in names:
            if name not in AVAILABLE_FILTERS:
                raise ValueError(
                    f"Unsupported filter '{value}'. Choose from: {' '.join(AVAILABLE_FILTERS)} or all."
                )
            if name not in expanded:
                expanded.append(name)
    return expanded


class FilterToggleController:
    def __init__(self, *, all_filter_names: list[str], initially_enabled: list[str]) -> None:
        self.all_filter_names = list(all_filter_names)
        self._enabled = {name: name in initially_enabled for name in self.all_filter_names}
        if any(self._enabled.get(name, False) for name in self.all_filter_names if name != "none"):
            self._enabled["none"] = False
        self._lock = threading.Lock()
        self._last_command_message = self.status_summary()

    def _enabled_names_locked(self) -> list[str]:
        return [name for name in self.all_filter_names if self._enabled.get(name, False)]

    def is_enabled(self, name: str) -> bool:
        with self._lock:
            return self._enabled.get(name, False)

    def enabled_names(self) -> list[str]:
        with self._lock:
            return self._enabled_names_locked()

    def status_summary(self) -> str:
        enabled = self.enabled_names()
        return f"Active filters: {', '.join(enabled) if enabled else 'off'}"

    def last_command_message(self) -> str:
        with self._lock:
            return self._last_command_message

    def set_enabled(self, name: str, enabled: bool) -> str:
        with self._lock:
            if enabled:
                if name == "none":
                    for filter_name in self.all_filter_names:
                        self._enabled[filter_name] = filter_name == "none"
                else:
                    self._enabled[name] = True
                    self._enabled["none"] = False
            else:
                self._enabled[name] = False

            self._enabled[name] = enabled
            state = "ON" if enabled else "OFF"
            self._last_command_message = (
                f"{name} {state} | "
                f"{', '.join(self._enabled_names_locked()) or 'off'}"
            )
            return self._last_command_message

    def set_all(self, enabled: bool) -> str:
        with self._lock:
            if enabled:
                for name in self.all_filter_names:
                    self._enabled[name] = name != "none"
            else:
                for name in self.all_filter_names:
                    self._enabled[name] = False
            state = "ON" if enabled else "OFF"
            self._last_command_message = f"all {state} | {', '.join(self._enabled_names_locked()) or 'off'}"
            return self._last_command_message

    def _set_all_locked(self, enabled: bool) -> None:
        if enabled:
            for name in self.all_filter_names:
                self._enabled[name] = name != "none"
        else:
            for name in self.all_filter_names:
                self._enabled[name] = False

    def handle_command(self, command_line: str) -> str:
        tokens = command_line.strip().lower().split()
        if not tokens:
            return self.last_command_message()

        if tokens[0] in {"help", "h", "?"}:
            return (
                "Commands: c|complementary on/off, k|kalman on/off, "
                "mh|mahony on/off, mg|madgwick on/off, n|none on/off, "
                "all on/off, status, help. One line can contain multiple pairs such as "
                "'c on k on'. If a line contains any ON pair, filters not explicitly turned "
                "on in that same line are turned off first."
            )

        if tokens[0] in {"status", "s"}:
            return self.status_summary()

        if len(tokens) % 2 != 0:
            return "Invalid command. Use help to see supported runtime filter commands."

        pairs: list[tuple[str, bool]] = []
        for index in range(0, len(tokens), 2):
            alias = tokens[index]
            action = tokens[index + 1]
            if action not in {"on", "off"}:
                return "Invalid command. Use help to see supported runtime filter commands."

            filter_name = FILTER_COMMAND_ALIASES.get(alias)
            if filter_name is None:
                return f"Unknown filter alias '{alias}'. Use help to see supported commands."
            pairs.append((filter_name, action == "on"))

        if any(filter_name == "all" for filter_name, _enabled in pairs) and len(pairs) > 1:
            return "Use 'all on' or 'all off' by itself, not mixed with other filter pairs."

        if len(pairs) == 1 and pairs[0][0] == "all":
            return self.set_all(pairs[0][1])

        contains_on = any(enabled for _filter_name, enabled in pairs)
        with self._lock:
            if contains_on:
                self._set_all_locked(False)

            for filter_name, enabled in pairs:
                if enabled:
                    if filter_name == "none":
                        self._set_all_locked(False)
                        self._enabled["none"] = True
                    else:
                        self._enabled[filter_name] = True
                        self._enabled["none"] = False
                else:
                    self._enabled[filter_name] = False

            pair_summary = " ".join(
                f"{filter_name} {'ON' if enabled else 'OFF'}"
                for filter_name, enabled in pairs
            )
            self._last_command_message = (
                f"{pair_summary} | {', '.join(self._enabled_names_locked()) or 'off'}"
            )
            return self._last_command_message


class CommandInputThread(threading.Thread):
    def __init__(self, *, controller: FilterToggleController) -> None:
        super().__init__(daemon=True)
        self.controller = controller

    def run(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return

            if line == "":
                return

            message = self.controller.handle_command(line)
            print(message)


def resolve_figure_path(path_arg: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir.parent / "filter_results" / "experiment_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = Path(path_arg)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".png")

    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    return output_dir / candidate


def resolve_capture_paths(csv_arg: str | None, figure_arg: str | None) -> tuple[Path, Path]:
    if csv_arg and figure_arg:
        return (resolve_csv_path(csv_arg), resolve_figure_path(figure_arg))

    timestamp_stem = datetime.now().strftime("imu_relative_pose_%Y%m%d_%H%M%S")
    csv_path = resolve_csv_path(csv_arg or f"{timestamp_stem}.csv")
    figure_path = resolve_figure_path(figure_arg or f"{timestamp_stem}.png")
    return (csv_path, figure_path)


def autoscale_axis(ax, series_list: list[list[float]]) -> None:
    values = [value for series in series_list for value in series]
    if not values:
        return

    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.1, 1.0)
    else:
        pad = max((high - low) * 0.15, 1.0)

    ax.set_ylim(low - pad, high + pad)


def angle_to_display_deg(angle_deg: float) -> float:
    return max(-90.0, min(90.0, float(angle_deg)))


def run_board_calibration(serial_port, timeout_s: float) -> None:
    print("Sending command: CALIBRATE")
    send_command(serial_port, "CALIBRATE")
    deadline = time.monotonic() + max(timeout_s, 1.0)
    completion_seen = False

    while time.monotonic() < deadline:
        raw = serial_port.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        print(line)

        if "# Calibration complete and saved to EEPROM." in line:
            completion_seen = True
            continue

        fields = [part.strip() for part in line.split(",")]
        if completion_seen and classify_header(fields):
            print("Calibration finished and CSV header resumed.")
            return

        if completion_seen and parse_data_line(line) is not None:
            print("Calibration finished and data stream resumed.")
            return

    raise TimeoutError("Timed out waiting for CALIBRATE to finish and resume streaming.")


def build_csv_fields(filter_names: list[str]) -> list[str]:
    fields = ["millis", "time_s", "phase", "baseline_ready", *SENSOR_FIELDS]
    for name in filter_names:
        fields.append(f"{name}_roll_deg")
        fields.append(f"{name}_pitch_deg")
        fields.append(f"{name}_yaw_deg")
    return fields


def build_csv_row(
    millis: float,
    phase: str,
    sensor_row: dict[str, float],
    estimates: dict[str, PoseEstimate] | None,
    filter_names: list[str],
    baseline_ready: bool,
) -> dict[str, str]:
    row = {
        "millis": str(int(round(millis))),
        "time_s": f"{(millis / 1000.0):.3f}",
        "phase": phase,
        "baseline_ready": "1" if baseline_ready else "0",
    }
    for field in SENSOR_FIELDS:
        row[field] = f"{sensor_row[field]:.6f}"

    for name in filter_names:
        if estimates is None or name not in estimates:
            row[f"{name}_roll_deg"] = ""
            row[f"{name}_pitch_deg"] = ""
            row[f"{name}_yaw_deg"] = ""
            continue

        estimate = estimates[name]
        row[f"{name}_roll_deg"] = f"{estimate.roll_deg:.6f}"
        row[f"{name}_pitch_deg"] = f"{estimate.pitch_deg:.6f}"
        row[f"{name}_yaw_deg"] = f"{estimate.yaw_deg:.6f}"

    return row


class RelativePoseTracker:
    def __init__(
        self,
        *,
        filters: dict[str, BasePoseFilter],
        baseline_seconds: float,
        low_pass_alpha: float,
    ) -> None:
        self.filters = filters
        self.filter_names = list(filters.keys())
        self.baseline_seconds = max(baseline_seconds, 0.5)
        self.low_pass = VectorLowPassFilter(alpha=low_pass_alpha)
        self.previous_millis: float | None = None
        self.baseline_start_millis: float | None = None
        self.baseline_samples = {name: [] for name in self.filter_names}
        self.baselines: dict[str, tuple[float, float, float, float]] = {}
        self.baseline_ready = False
        self.last_status = "Waiting for baseline samples..."

    @property
    def phase(self) -> str:
        return "tracking" if self.baseline_ready else "baseline"

    def process(
        self,
        measurement_row: dict[str, float],
    ) -> tuple[float, dict[str, float], dict[str, PoseEstimate] | None, str, bool]:
        millis = float(measurement_row["millis"])
        if self.previous_millis is None:
            dt_s = 0.333
        else:
            dt_s = max(0.001, min((millis - self.previous_millis) / 1000.0, 1.0))
        self.previous_millis = millis

        accel = (
            float(measurement_row["ax_mps2"]),
            float(measurement_row["ay_mps2"]),
            float(measurement_row["az_mps2"]),
        )
        gyro = (
            float(measurement_row["gx_dps"]),
            float(measurement_row["gy_dps"]),
            float(measurement_row["gz_dps"]),
        )

        accel = self.low_pass.apply("accel", accel)
        gyro = self.low_pass.apply("gyro", gyro)
        sensor_row = {
            "ax_mps2": accel[0],
            "ay_mps2": accel[1],
            "az_mps2": accel[2],
            "gx_dps": gyro[0],
            "gy_dps": gyro[1],
            "gz_dps": gyro[2],
        }

        absolute_estimates: dict[str, PoseEstimate] = {}
        for name, pose_filter in self.filters.items():
            quaternion = pose_filter.update(accel, gyro, dt_s)
            absolute_estimates[name] = pose_estimate_from_quaternion(quaternion)

        if not self.baseline_ready:
            if self.baseline_start_millis is None:
                self.baseline_start_millis = millis

            for name, estimate in absolute_estimates.items():
                self.baseline_samples[name].append(estimate.quaternion)

            elapsed_s = (millis - self.baseline_start_millis) / 1000.0
            remaining_s = max(0.0, self.baseline_seconds - elapsed_s)
            self.last_status = f"Capturing baseline pose... keep still for {remaining_s:.1f}s"

            if elapsed_s >= self.baseline_seconds:
                for name, samples in self.baseline_samples.items():
                    self.baselines[name] = average_quaternions(samples)
                self.baseline_ready = True
                self.last_status = (
                    f"Baseline locked from {len(self.baseline_samples[self.filter_names[0]])} samples"
                )
                return (
                    millis,
                    sensor_row,
                    self._relative_estimates(absolute_estimates),
                    self.last_status,
                    True,
                )

            return (millis, sensor_row, None, self.last_status, False)

        self.last_status = "Tracking relative pose"
        return (
            millis,
            sensor_row,
            self._relative_estimates(absolute_estimates),
            self.last_status,
            False,
        )

    def _relative_estimates(
        self,
        absolute_estimates: dict[str, PoseEstimate],
    ) -> dict[str, PoseEstimate]:
        relative_estimates: dict[str, PoseEstimate] = {}
        for name, estimate in absolute_estimates.items():
            baseline = self.baselines[name]
            relative_estimates[name] = pose_estimate_from_quaternion(
                relative_quaternion(baseline, estimate.quaternion)
            )
        return relative_estimates


class LiveBuffer:
    def __init__(self, *, window_seconds: float, filter_names: list[str]) -> None:
        self.window_seconds = max(window_seconds, 1.0)
        self.filter_names = filter_names
        self.timestamps = deque()
        self.sensor_values = {field: deque() for field in SENSOR_FIELDS}
        self.angle_values = {
            name: {axis: deque() for axis in PLOT_AXES}
            for name in filter_names
        }
        self.history_timestamps: list[float] = []
        self.history_sensor_values = {field: [] for field in SENSOR_FIELDS}
        self.history_angle_values = {
            name: {axis: [] for axis in PLOT_AXES}
            for name in filter_names
        }
        self.sample_count = 0
        self.last_message = "Waiting for IMU data..."
        self._lock = threading.Lock()

    def clear_data(self) -> None:
        with self._lock:
            self.timestamps.clear()
            for series in self.sensor_values.values():
                series.clear()
            for filter_values in self.angle_values.values():
                for series in filter_values.values():
                    series.clear()

    def add_sample(
        self,
        *,
        timestamp_s: float,
        sensor_row: dict[str, float],
        estimates: dict[str, PoseEstimate] | None,
        message: str,
    ) -> None:
        with self._lock:
            self.timestamps.append(timestamp_s)
            for field in SENSOR_FIELDS:
                self.sensor_values[field].append(sensor_row[field])
                self.history_sensor_values[field].append(sensor_row[field])
            self.history_timestamps.append(timestamp_s)
            for name in self.filter_names:
                if estimates is None or name not in estimates:
                    self.angle_values[name]["roll_deg"].append(float("nan"))
                    self.angle_values[name]["pitch_deg"].append(float("nan"))
                    self.angle_values[name]["yaw_deg"].append(float("nan"))
                    self.history_angle_values[name]["roll_deg"].append(float("nan"))
                    self.history_angle_values[name]["pitch_deg"].append(float("nan"))
                    self.history_angle_values[name]["yaw_deg"].append(float("nan"))
                    continue

                estimate = estimates[name]
                roll_display = angle_to_display_deg(estimate.roll_deg)
                pitch_display = angle_to_display_deg(estimate.pitch_deg)
                yaw_display = angle_to_display_deg(estimate.yaw_deg)
                self.angle_values[name]["roll_deg"].append(roll_display)
                self.angle_values[name]["pitch_deg"].append(pitch_display)
                self.angle_values[name]["yaw_deg"].append(yaw_display)
                self.history_angle_values[name]["roll_deg"].append(roll_display)
                self.history_angle_values[name]["pitch_deg"].append(pitch_display)
                self.history_angle_values[name]["yaw_deg"].append(yaw_display)
            self.sample_count += 1
            self.last_message = message
            self._trim_locked()

    def set_message(self, message: str) -> None:
        with self._lock:
            self.last_message = message

    def snapshot(
        self,
    ) -> tuple[list[float], dict[str, list[float]], dict[str, dict[str, list[float]]], int, str]:
        with self._lock:
            return (
                list(self.timestamps),
                {field: list(series) for field, series in self.sensor_values.items()},
                {
                    name: {axis: list(series) for axis, series in filter_values.items()}
                    for name, filter_values in self.angle_values.items()
                },
                self.sample_count,
                self.last_message,
            )

    def history_snapshot(
        self,
    ) -> tuple[list[float], dict[str, list[float]], dict[str, dict[str, list[float]]], int, str]:
        with self._lock:
            return (
                list(self.history_timestamps),
                {field: list(series) for field, series in self.history_sensor_values.items()},
                {
                    name: {axis: list(series) for axis, series in filter_values.items()}
                    for name, filter_values in self.history_angle_values.items()
                },
                self.sample_count,
                self.last_message,
            )

    def _trim_locked(self) -> None:
        if not self.timestamps:
            return

        cutoff = self.timestamps[-1] - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            for series in self.sensor_values.values():
                series.popleft()
            for filter_values in self.angle_values.values():
                for axis in PLOT_AXES:
                    filter_values[axis].popleft()


class SerialReader(threading.Thread):
    def __init__(
        self,
        *,
        serial_port,
        tracker: RelativePoseTracker,
        buffer: LiveBuffer,
        filter_names: list[str],
        writer: csv.DictWriter | None = None,
        csv_file=None,
    ) -> None:
        super().__init__(daemon=True)
        self.serial_port = serial_port
        self.tracker = tracker
        self.buffer = buffer
        self.filter_names = filter_names
        self.writer = writer
        self.csv_file = csv_file
        self.stop_event = threading.Event()
        self.error: str | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        flush_every = 25
        pending_rows = 0

        try:
            while not self.stop_event.is_set():
                raw = self.serial_port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("#"):
                    print(line)
                    self.buffer.set_message(line)
                    continue

                parsed = parse_data_line(line)
                if parsed is None:
                    print(f"Skipping malformed line: {line}")
                    continue

                measurement_row = serial_row_to_measurement_row(parsed)
                millis, sensor_row, estimates, message, baseline_locked = self.tracker.process(
                    measurement_row
                )

                if baseline_locked:
                    print("Baseline capture complete. Relative pose tracking started.")
                    self.buffer.clear_data()
                    self.buffer.set_message(message)
                else:
                    self.buffer.add_sample(
                        timestamp_s=millis / 1000.0,
                        sensor_row=sensor_row,
                        estimates=estimates,
                        message=message,
                    )

                if self.writer is not None:
                    self.writer.writerow(
                        build_csv_row(
                            millis=millis,
                            phase=self.tracker.phase,
                            sensor_row=sensor_row,
                            estimates=estimates,
                            filter_names=self.filter_names,
                            baseline_ready=self.tracker.baseline_ready,
                        )
                    )
                    pending_rows += 1
                    if pending_rows >= flush_every and self.csv_file is not None:
                        self.csv_file.flush()
                        pending_rows = 0
        except Exception as exc:
            self.error = f"Serial reader stopped: {exc}"
        finally:
            if self.csv_file is not None:
                self.csv_file.flush()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    if not args.port:
        print("--port is required unless --list-ports is used.", file=sys.stderr)
        return 2

    try:
        initial_active_filter_names = expand_filter_names(args.filters)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    all_filter_names = list(AVAILABLE_FILTERS)
    toggle_controller = FilterToggleController(
        all_filter_names=all_filter_names,
        initially_enabled=initial_active_filter_names,
    )

    low_pass_alpha = 1.0 if args.disable_low_pass else args.low_pass_alpha
    low_pass_alpha = max(0.0, min(low_pass_alpha, 1.0))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required. Install it with: pip install matplotlib", file=sys.stderr)
        return 1

    filters = {
        name: build_pose_filter(
            name,
            complementary_alpha=args.complementary_alpha,
            kalman_q_angle=args.kalman_q_angle,
            kalman_q_bias=args.kalman_q_bias,
            kalman_r_measure=args.kalman_r_measure,
            mahony_kp=args.mahony_kp,
            mahony_ki=args.mahony_ki,
            madgwick_beta=args.madgwick_beta,
        )
        for name in all_filter_names
    }

    tracker = RelativePoseTracker(
        filters=filters,
        baseline_seconds=args.baseline_seconds,
        low_pass_alpha=low_pass_alpha,
    )
    buffer = LiveBuffer(window_seconds=args.window_seconds, filter_names=all_filter_names)

    csv_file = None
    writer = None
    csv_path: Path | None = None
    figure_path: Path | None = None
    reader = None
    save_figure_snapshot_callback: Callable[[], None] | None = None
    if args.no_save:
        if args.csv or args.save_figure:
            print("Ignoring --csv and --save-figure because --no-save was requested.")
    else:
        csv_path, figure_path = resolve_capture_paths(args.csv, args.save_figure)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=build_csv_fields(all_filter_names))
        writer.writeheader()
        csv_file.flush()

    low_pass_text = "off" if low_pass_alpha >= 1.0 else f"on (alpha={low_pass_alpha:.2f})"

    try:
        with open_serial(args.port, args.baud) as serial_port:
            print(f"Connected to {args.port} @ {args.baud}")
            print(f"Selected filters: {', '.join(initial_active_filter_names)}")
            print(f"Low-pass preprocessing: {low_pass_text}")
            print("Yaw is gyro-only in this 6-axis setup and will drift over time.")
            print("Runtime filter commands: c/k/mh/mg/n on|off, all on|off, status, help")

            if csv_path is not None and figure_path is not None:
                print(f"Saving processed CSV to {csv_path}")
                print(f"Relative-pose image will be saved to {figure_path} when the window closes")
            else:
                print("Capture saving disabled (--no-save).")

            if args.startup_delay > 0:
                print(f"Waiting {args.startup_delay:.1f}s for board startup...")
                time.sleep(args.startup_delay)

            wait_for_header(serial_port, send_imu=False, timeout_s=args.header_timeout)

            if args.calibrate_first:
                run_board_calibration(serial_port, timeout_s=args.calibration_timeout)

            print(
                f"Hold the IMU in the desired baseline pose for {args.baseline_seconds:.1f}s "
                "while the baseline is captured."
            )

            reader = SerialReader(
                serial_port=serial_port,
                tracker=tracker,
                buffer=buffer,
                filter_names=all_filter_names,
                writer=writer,
                csv_file=csv_file,
            )
            reader.start()

            input_thread = None
            if sys.stdin is not None and sys.stdin.isatty():
                input_thread = CommandInputThread(controller=toggle_controller)
                input_thread.start()
            else:
                print("Runtime filter toggles are unavailable because stdin is not interactive.")

            figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
            figure.suptitle("MPU6050 Relative Pose")

            accel_lines = {
                field: axes[0].plot([], [], label=field, color=ACCEL_COLORS[field])[0]
                for field in ACCEL_UNIT_FIELDS
            }
            gyro_lines = {
                field: axes[1].plot([], [], label=field, color=GYRO_COLORS[field])[0]
                for field in GYRO_UNIT_FIELDS
            }
            angle_lines = {
                name: {
                    axis: axes[2].plot(
                        [],
                        [],
                        label=f"{name}:{axis.replace('_deg', '')}",
                        color=FILTER_COLORS.get(name, None),
                        linestyle=ANGLE_LINE_STYLES[axis],
                    )[0]
                    for axis in PLOT_AXES
                }
                for name in all_filter_names
            }

            axes[0].set_ylabel("Accel (m/s^2)")
            axes[1].set_ylabel("Gyro (deg/s)")
            axes[2].set_ylabel("Angle (deg)")
            axes[2].set_ylim(-90.0, 90.0)

            for axis in axes:
                axis.grid(True, alpha=0.25)
                axis.set_xlim(0, args.window_seconds)
            axes[-1].set_xlabel("Time in Window (s)")
            axes[0].legend(loc="upper left")
            axes[1].legend(loc="upper left")
            axes[2].legend(loc="upper left", ncol=3, fontsize=8)

            status_text = figure.text(
                0.01,
                0.98,
                "Waiting for IMU data...",
                va="top",
            )
            figure_saved = False
            figure_save_lock = threading.Lock()

            def update(_frame):
                timestamps, sensor_values, angle_values, sample_count, last_message = buffer.snapshot()
                artists = [status_text]
                artists.extend(accel_lines.values())
                artists.extend(gyro_lines.values())
                for filter_lines in angle_lines.values():
                    artists.extend(filter_lines.values())

                if not timestamps:
                    status_text.set_text(
                        f"{last_message} | {toggle_controller.status_summary()}"
                    )
                    return artists

                newest_time = timestamps[-1]
                oldest_time = timestamps[0]
                if (newest_time - oldest_time) < args.window_seconds:
                    window_start = oldest_time
                else:
                    window_start = newest_time - args.window_seconds
                x_values = [timestamp - window_start for timestamp in timestamps]
                active_filter_names = toggle_controller.enabled_names()

                for field, line in accel_lines.items():
                    line.set_data(x_values, sensor_values[field])
                for field, line in gyro_lines.items():
                    line.set_data(x_values, sensor_values[field])

                for name in all_filter_names:
                    visible = name in active_filter_names
                    for axis_name in PLOT_AXES:
                        line = angle_lines[name][axis_name]
                        line.set_visible(visible)
                        if visible:
                            line.set_data(x_values, angle_values[name][axis_name])
                        else:
                            line.set_data([], [])

                for axis in axes:
                    axis.set_xlim(0, args.window_seconds)

                autoscale_axis(axes[0], [sensor_values[field] for field in ACCEL_UNIT_FIELDS])
                autoscale_axis(axes[1], [sensor_values[field] for field in GYRO_UNIT_FIELDS])
                axes[2].set_ylim(-90.0, 90.0)

                status_text.set_text(
                    f"Samples: {sample_count} | Latest board time: {newest_time:.2f}s | "
                    f"{last_message} | Angles shown as -90 to +90 deg | {toggle_controller.status_summary()} | "
                    f"Last cmd: {toggle_controller.last_command_message()}"
                )
                return artists

            def save_figure_snapshot() -> None:
                nonlocal figure_saved
                if figure_path is None:
                    return
                with figure_save_lock:
                    if figure_saved:
                        return

                    (
                        history_timestamps,
                        history_sensor_values,
                        history_angle_values,
                        history_sample_count,
                        _history_message,
                    ) = buffer.history_snapshot()
                    if not history_timestamps:
                        update(None)
                        figure.savefig(figure_path, dpi=150)
                        figure_saved = True
                        print(f"Saved relative-pose image to {figure_path}")
                        return

                    history_figure, history_axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
                    history_figure.suptitle("MPU6050 Relative Pose - Full Session")

                    x_values = [timestamp - history_timestamps[0] for timestamp in history_timestamps]
                    active_filter_names = toggle_controller.enabled_names()

                    for field in ACCEL_UNIT_FIELDS:
                        history_axes[0].plot(
                            x_values,
                            history_sensor_values[field],
                            label=field,
                            color=ACCEL_COLORS[field],
                        )
                    for field in GYRO_UNIT_FIELDS:
                        history_axes[1].plot(
                            x_values,
                            history_sensor_values[field],
                            label=field,
                            color=GYRO_COLORS[field],
                        )
                    for name in active_filter_names:
                        for axis_name in PLOT_AXES:
                            history_axes[2].plot(
                                x_values,
                                history_angle_values[name][axis_name],
                                label=f"{name}:{axis_name.replace('_deg', '')}",
                                color=FILTER_COLORS.get(name, None),
                                linestyle=ANGLE_LINE_STYLES[axis_name],
                            )

                    history_axes[0].set_ylabel("Accel (m/s^2)")
                    history_axes[1].set_ylabel("Gyro (deg/s)")
                    history_axes[2].set_ylabel("Angle (deg)")
                    history_axes[2].set_xlabel("Time Since Start (s)")
                    history_axes[2].set_ylim(-90.0, 90.0)

                    for axis in history_axes:
                        axis.grid(True, alpha=0.25)

                    autoscale_axis(
                        history_axes[0],
                        [history_sensor_values[field] for field in ACCEL_UNIT_FIELDS],
                    )
                    autoscale_axis(
                        history_axes[1],
                        [history_sensor_values[field] for field in GYRO_UNIT_FIELDS],
                    )

                    history_axes[0].legend(loc="upper left")
                    history_axes[1].legend(loc="upper left")
                    if active_filter_names:
                        history_axes[2].legend(loc="upper left", ncol=3, fontsize=8)

                    duration_s = max(0.0, history_timestamps[-1] - history_timestamps[0])
                    history_figure.text(
                        0.01,
                        0.98,
                        (
                            f"Samples: {history_sample_count} | Duration: {duration_s:.2f}s | "
                            f"Angles shown as -90 to +90 deg | Saved filters: "
                            f"{', '.join(active_filter_names) if active_filter_names else 'none'}"
                        ),
                        va="top",
                    )
                    history_figure.tight_layout(rect=(0, 0, 1, 0.95))
                    history_figure.savefig(figure_path, dpi=150)
                    plt.close(history_figure)
                    figure_saved = True
                    print(f"Saved relative-pose image to {figure_path}")

            save_figure_snapshot_callback = save_figure_snapshot

            def handle_close(_event) -> None:
                reader.stop()
                reader.join(timeout=2.0)
                save_figure_snapshot()

            figure.canvas.mpl_connect("close_event", handle_close)
            animation = FuncAnimation(
                figure,
                update,
                interval=args.refresh_ms,
                cache_frame_data=False,
            )
            figure._animation = animation
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            plt.show()

            reader.stop()
            reader.join(timeout=2.0)

            save_figure_snapshot()

            if reader.error is not None:
                print(reader.error, file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        if reader is not None:
            reader.stop()
            reader.join(timeout=2.0)
        if save_figure_snapshot_callback is not None:
            save_figure_snapshot_callback()
        print("Relative-pose viewer interrupted by user.")
    except TimeoutError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
