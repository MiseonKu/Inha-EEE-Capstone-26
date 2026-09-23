from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


VIEWER_FIELDS = (
    "millis",
    "mode",
    "estimate_valid",
    "elbow_primary_deg",
    "elbow_backup_deg",
    "elbow_fused_deg",
    "upper_tilt_deg",
    "forearm_tilt_deg",
    "upper_consistency_deg",
    "forearm_dynamic_penalty_mps2",
    "upper_confidence",
    "forearm_confidence",
    "upper_dir_x",
    "upper_dir_y",
    "upper_dir_z",
    "forearm_dir_x",
    "forearm_dir_y",
    "forearm_dir_z",
    "upper_hinge_x",
    "upper_hinge_y",
    "upper_hinge_z",
    "forearm_hinge_x",
    "forearm_hinge_y",
    "forearm_hinge_z",
    "u1_valid",
    "u1_ax_mps2",
    "u1_ay_mps2",
    "u1_az_mps2",
    "u1_gx_dps",
    "u1_gy_dps",
    "u1_gz_dps",
    "u2_valid",
    "u2_ax_mps2",
    "u2_ay_mps2",
    "u2_az_mps2",
    "u2_gx_dps",
    "u2_gy_dps",
    "u2_gz_dps",
    "f1_valid",
    "f1_ax_mps2",
    "f1_ay_mps2",
    "f1_az_mps2",
    "f1_gx_dps",
    "f1_gy_dps",
    "f1_gz_dps",
)

MODE_NAMES = {
    0: "idle",
    1: "reference-cal",
    2: "axis-cal",
    3: "run",
}

SENSOR_OPTIONS = {
    "1": ("u1", "Upper IMU 1"),
    "2": ("u2", "Upper IMU 2"),
    "3": ("f1", "Forearm IMU 1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaF Unit 1 IMU live validation viewer"
    )
    parser.add_argument("--port", help="Serial port such as COM12")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument("--demo", action="store_true", help="Run without serial using synthetic motion")
    parser.add_argument("--window-seconds", type=float, default=12.0, help="Visible time window in seconds")
    parser.add_argument("--refresh-ms", type=int, default=70, help="Screen refresh interval in milliseconds")
    parser.add_argument("--startup-delay", type=float, default=2.0, help="Seconds to wait after opening the serial port")
    parser.add_argument("--header-timeout", type=float, default=15.0, help="Maximum seconds to wait for the viewer header")
    parser.add_argument("--upper-length", type=float, default=0.28937, help="Upper-arm link length in meters")
    parser.add_argument("--forearm-length", type=float, default=0.20000, help="Forearm link length in meters")
    parser.add_argument("--base-height", type=float, default=0.07002, help="Base-to-shoulder height in meters")
    parser.add_argument("--csv", help="Optional CSV file to save parsed viewer frames")
    parser.add_argument("--save-figure", help="Optional figure image path to save when the window closes")
    return parser.parse_args()


def require_numpy() -> None:
    if np is None:
        print("numpy is required. pip install numpy", file=sys.stderr)
        raise SystemExit(1)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required. pip install matplotlib", file=sys.stderr)
        raise SystemExit(1)
    return plt, FuncAnimation


def require_serial() -> None:
    if serial is None:
        print("pyserial is required. pip install pyserial", file=sys.stderr)
        raise SystemExit(1)


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-9:
        return np.zeros_like(vector)
    return vector / vector_norm


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rotate_about_axis(vector, axis, angle_rad: float):
    axis = normalize(axis)
    vector = np.asarray(vector, dtype=float)
    if float(np.linalg.norm(axis)) < 1.0e-9:
        return vector.copy()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        vector * c
        + np.cross(axis, vector) * s
        + axis * np.dot(axis, vector) * (1.0 - c)
    )


def sensor_rotation_from_link_axes(link_direction_world, hinge_axis_world, *, upper_link: bool):
    hinge_axis_world = normalize(hinge_axis_world)
    link_direction_world = normalize(link_direction_world)

    if upper_link:
        z_axis = link_direction_world
        y_axis = hinge_axis_world
        x_axis = normalize(np.cross(y_axis, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
    else:
        x_axis = link_direction_world
        y_axis = hinge_axis_world
        z_axis = normalize(np.cross(x_axis, y_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))

    return np.column_stack((x_axis, y_axis, z_axis))


def classify_header(parts: List[str]) -> bool:
    return parts == ["VH", *VIEWER_FIELDS]


def parse_viewer_line(line: str) -> Optional[Dict[str, float]]:
    parts = [part.strip() for part in line.split(",")]
    if not parts or classify_header(parts):
        return None

    if len(parts) != len(VIEWER_FIELDS) + 1 or parts[0] != "VD":
        return None

    numeric_values = parts[1:]
    try:
        return {
            field: float(value)
            for field, value in zip(VIEWER_FIELDS, numeric_values)
        }
    except ValueError:
        return None


def resolve_output_path(path_arg: str, suffix: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    candidate = Path(path_arg)

    if candidate.suffix == "":
        candidate = candidate.with_suffix(suffix)

    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    output_path = base_dir / candidate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def list_ports() -> None:
    require_serial()
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        description = port.description or "Unknown device"
        print(f"{port.device}: {description}")


def open_serial(port: str, baud: int):
    require_serial()
    serial_port = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=1.0,
        rtscts=False,
        dsrdtr=False,
    )

    try:
        serial_port.setDTR(False)
        serial_port.setRTS(False)
    except OSError:
        pass

    return serial_port


def send_command(serial_port, command: str) -> None:
    serial_port.write((command.strip() + "\n").encode("utf-8"))
    serial_port.flush()


def wait_for_viewer_header(serial_port, timeout_s: float) -> None:
    print("Waiting for viewer header from board...")
    start_time = time.monotonic()
    last_request_time = 0.0

    while True:
        now = time.monotonic()
        if timeout_s > 0.0 and now - start_time >= timeout_s:
            raise TimeoutError(
                "Timed out waiting for the viewer header. Check demo.ino upload and serial wiring."
            )

        if now - last_request_time >= 1.0:
            send_command(serial_port, "p")
            last_request_time = now

        raw = serial_port.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if line.startswith("#"):
            print(line)
            continue

        parts = [part.strip() for part in line.split(",")]
        if classify_header(parts):
            print("Viewer header received.")
            return

        if parse_viewer_line(line) is not None:
            print("Viewer data arrived before a fresh header. Continuing.")
            return


class LiveBuffer:
    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = max(window_seconds, 2.0)
        self.rows: Deque[Dict[str, float]] = deque()
        self.sample_count = 0
        self.last_message = "waiting"
        self.first_timestamp_s: Optional[float] = None
        self._lock = threading.Lock()

    def add_row(self, row: Dict[str, float]) -> None:
        timestamp_s = row["millis"] / 1000.0

        with self._lock:
            self.rows.append(dict(row))
            self.sample_count += 1
            if self.first_timestamp_s is None:
                self.first_timestamp_s = timestamp_s
            self.last_message = "streaming"
            self._trim_locked(timestamp_s)

    def set_message(self, message: str) -> None:
        with self._lock:
            self.last_message = message

    def snapshot(self) -> Tuple[List[Dict[str, float]], int, str, Optional[float]]:
        with self._lock:
            return (
                list(self.rows),
                self.sample_count,
                self.last_message,
                self.first_timestamp_s,
            )

    def _trim_locked(self, newest_timestamp_s: float) -> None:
        cutoff = newest_timestamp_s - self.window_seconds
        while self.rows and (self.rows[0]["millis"] / 1000.0) < cutoff:
            self.rows.popleft()


class SerialProducer(threading.Thread):
    def __init__(
        self,
        serial_port,
        buffer: LiveBuffer,
        writer: Optional[csv.DictWriter] = None,
        csv_file=None,
    ) -> None:
        super().__init__(daemon=True)
        self.serial_port = serial_port
        self.buffer = buffer
        self.writer = writer
        self.csv_file = csv_file
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self._write_lock = threading.Lock()

    def stop(self) -> None:
        self.stop_event.set()

    def send(self, command: str) -> None:
        with self._write_lock:
            send_command(self.serial_port, command)

    def run(self) -> None:
        pending_flush_rows = 0

        try:
            while not self.stop_event.is_set():
                raw = self.serial_port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("#"):
                    self.buffer.set_message(line[2:] if line.startswith("# ") else line)
                    continue

                row = parse_viewer_line(line)
                if row is None:
                    continue

                self.buffer.add_row(row)

                if self.writer is not None:
                    self.writer.writerow(
                        {field: f"{row[field]:.6f}" for field in VIEWER_FIELDS}
                    )
                    pending_flush_rows += 1
                    if pending_flush_rows >= 20 and self.csv_file is not None:
                        self.csv_file.flush()
                        pending_flush_rows = 0
        except Exception as exc:
            self.error = f"시리얼 수신 스레드 중단: {exc}"
        finally:
            if self.csv_file is not None:
                self.csv_file.flush()


class DemoProducer(threading.Thread):
    def __init__(self, buffer: LiveBuffer) -> None:
        super().__init__(daemon=True)
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.start_time = time.monotonic()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            elapsed_s = time.monotonic() - self.start_time
            self.buffer.add_row(self.make_demo_row(elapsed_s))
            self.buffer.set_message("demo mode running")
            time.sleep(0.05)

    # 3D 화면과 파형이 어떤 모습으로 나오는지 바로 확인할 수 있도록
    # 실제와 비슷한 형태의 합성 동작 데이터를 만든다.
    def make_demo_row(self, elapsed_s: float) -> Dict[str, float]:
        gravity_world = np.array([0.0, 0.0, 9.80665], dtype=float)

        upper_tilt_rad = math.radians(16.0 * math.sin(0.42 * elapsed_s))
        upper_azimuth_rad = 0.58 * math.sin(0.23 * elapsed_s)
        upper_dir = normalize(
            np.array(
                [
                    math.sin(upper_tilt_rad) * math.cos(upper_azimuth_rad),
                    math.sin(upper_tilt_rad) * math.sin(upper_azimuth_rad),
                    math.cos(upper_tilt_rad),
                ],
                dtype=float,
            )
        )

        hinge_axis = normalize(
            np.array(
                [-math.sin(upper_azimuth_rad), math.cos(upper_azimuth_rad), 0.0],
                dtype=float,
            )
        )
        if float(np.linalg.norm(hinge_axis)) < 1.0e-9:
            hinge_axis = np.array([0.0, 1.0, 0.0], dtype=float)

        forearm_reference = normalize(np.cross(hinge_axis, upper_dir))
        elbow_fused_deg = 42.0 * math.sin(0.78 * elapsed_s)
        elbow_primary_deg = elbow_fused_deg + 1.3 * math.sin(1.74 * elapsed_s)
        elbow_backup_deg = elbow_fused_deg - 1.8 * math.sin(1.21 * elapsed_s + 0.2)
        forearm_dir = normalize(
            rotate_about_axis(
                forearm_reference,
                hinge_axis,
                math.radians(-elbow_fused_deg),
            )
        )

        upper_rotation = sensor_rotation_from_link_axes(
            upper_dir,
            hinge_axis,
            upper_link=True,
        )
        forearm_rotation = sensor_rotation_from_link_axes(
            forearm_dir,
            hinge_axis,
            upper_link=False,
        )

        upper_accel = upper_rotation.T @ gravity_world
        forearm_accel = forearm_rotation.T @ gravity_world

        upper_accel += np.array(
            [
                0.18 * math.sin(2.30 * elapsed_s),
                0.14 * math.sin(1.85 * elapsed_s + 0.3),
                0.12 * math.sin(2.70 * elapsed_s + 0.1),
            ],
            dtype=float,
        )
        forearm_accel += np.array(
            [
                0.55 * math.sin(2.65 * elapsed_s + 0.2),
                0.35 * math.sin(1.90 * elapsed_s + 0.8),
                0.90 * math.sin(3.05 * elapsed_s + 0.5),
            ],
            dtype=float,
        )

        upper_gyro = np.array(
            [
                8.0 * math.sin(1.10 * elapsed_s),
                32.0 * math.cos(0.78 * elapsed_s),
                7.0 * math.sin(0.55 * elapsed_s + 0.2),
            ],
            dtype=float,
        )
        upper_gyro_2 = np.array(
            [
                9.5 * math.sin(1.08 * elapsed_s + 0.1),
                30.5 * math.cos(0.78 * elapsed_s + 0.06),
                7.4 * math.sin(0.54 * elapsed_s + 0.3),
            ],
            dtype=float,
        )
        forearm_gyro = np.array(
            [
                11.0 * math.sin(1.15 * elapsed_s + 0.4),
                68.0 * math.cos(0.78 * elapsed_s + 0.08),
                10.0 * math.sin(0.60 * elapsed_s + 0.7),
            ],
            dtype=float,
        )

        upper_consistency = 1.1 + 3.0 * (0.5 + 0.5 * abs(math.sin(1.22 * elapsed_s)))
        forearm_penalty = 0.30 + 2.40 * (0.5 + 0.5 * abs(math.sin(1.36 * elapsed_s + 0.2)))
        upper_conf = clamp(0.96 - 0.12 * abs(math.sin(1.35 * elapsed_s)), 0.30, 1.00)
        forearm_conf = clamp(0.90 - 0.28 * abs(math.sin(1.18 * elapsed_s + 0.4)), 0.20, 1.00)
        forearm_tilt_deg = math.degrees(math.asin(clamp(float(forearm_dir[2]), -1.0, 1.0)))
        upper_tilt_deg = math.degrees(
            math.acos(clamp(float(np.dot(upper_dir, np.array([0.0, 0.0, 1.0]))), -1.0, 1.0))
        )

        return {
            "millis": elapsed_s * 1000.0,
            "mode": 3.0,
            "estimate_valid": 1.0,
            "elbow_primary_deg": elbow_primary_deg,
            "elbow_backup_deg": elbow_backup_deg,
            "elbow_fused_deg": elbow_fused_deg,
            "upper_tilt_deg": upper_tilt_deg,
            "forearm_tilt_deg": forearm_tilt_deg,
            "upper_consistency_deg": upper_consistency,
            "forearm_dynamic_penalty_mps2": forearm_penalty,
            "upper_confidence": upper_conf,
            "forearm_confidence": forearm_conf,
            "upper_dir_x": float(upper_dir[0]),
            "upper_dir_y": float(upper_dir[1]),
            "upper_dir_z": float(upper_dir[2]),
            "forearm_dir_x": float(forearm_dir[0]),
            "forearm_dir_y": float(forearm_dir[1]),
            "forearm_dir_z": float(forearm_dir[2]),
            "upper_hinge_x": float(hinge_axis[0]),
            "upper_hinge_y": float(hinge_axis[1]),
            "upper_hinge_z": float(hinge_axis[2]),
            "forearm_hinge_x": float(hinge_axis[0]),
            "forearm_hinge_y": float(hinge_axis[1]),
            "forearm_hinge_z": float(hinge_axis[2]),
            "u1_valid": 1.0,
            "u1_ax_mps2": float(upper_accel[0]),
            "u1_ay_mps2": float(upper_accel[1]),
            "u1_az_mps2": float(upper_accel[2]),
            "u1_gx_dps": float(upper_gyro[0]),
            "u1_gy_dps": float(upper_gyro[1]),
            "u1_gz_dps": float(upper_gyro[2]),
            "u2_valid": 1.0,
            "u2_ax_mps2": float(upper_accel[0] + 0.11 * math.sin(1.80 * elapsed_s)),
            "u2_ay_mps2": float(upper_accel[1] - 0.08 * math.sin(2.10 * elapsed_s + 0.2)),
            "u2_az_mps2": float(upper_accel[2] + 0.10 * math.sin(2.00 * elapsed_s + 0.4)),
            "u2_gx_dps": float(upper_gyro_2[0]),
            "u2_gy_dps": float(upper_gyro_2[1]),
            "u2_gz_dps": float(upper_gyro_2[2]),
            "f1_valid": 1.0,
            "f1_ax_mps2": float(forearm_accel[0]),
            "f1_ay_mps2": float(forearm_accel[1]),
            "f1_az_mps2": float(forearm_accel[2]),
            "f1_gx_dps": float(forearm_gyro[0]),
            "f1_gy_dps": float(forearm_gyro[1]),
            "f1_gz_dps": float(forearm_gyro[2]),
        }


def autoscale_axis(axis, series_list: List[List[float]], *, default_half_range: float) -> None:
    merged = [value for series in series_list for value in series]
    if not merged:
        axis.set_ylim(-default_half_range, default_half_range)
        return

    low = min(merged)
    high = max(merged)
    if abs(high - low) < 1.0e-9:
        pad = max(default_half_range * 0.25, abs(high) * 0.15 + 0.5)
    else:
        pad = max((high - low) * 0.12, default_half_range * 0.10)
    axis.set_ylim(low - pad, high + pad)


def build_chain_points(row: Dict[str, float], base_height: float, upper_length: float, forearm_length: float):
    base = np.array([0.0, 0.0, 0.0], dtype=float)
    shoulder = np.array([0.0, 0.0, base_height], dtype=float)
    upper_dir = normalize(
        np.array(
            [row["upper_dir_x"], row["upper_dir_y"], row["upper_dir_z"]],
            dtype=float,
        )
    )
    forearm_dir = normalize(
        np.array(
            [row["forearm_dir_x"], row["forearm_dir_y"], row["forearm_dir_z"]],
            dtype=float,
        )
    )
    elbow = shoulder + upper_length * upper_dir
    wrist = elbow + forearm_length * forearm_dir
    return np.vstack((base, shoulder, elbow, wrist))


def reference_chain_points(base_height: float, upper_length: float, forearm_length: float):
    base = np.array([0.0, 0.0, 0.0], dtype=float)
    shoulder = np.array([0.0, 0.0, base_height], dtype=float)
    elbow = shoulder + np.array([0.0, 0.0, upper_length], dtype=float)
    wrist = elbow + np.array([forearm_length, 0.0, 0.0], dtype=float)
    return np.vstack((base, shoulder, elbow, wrist))


def axis_segment(origin, axis_vector, scale: float):
    axis_vector = normalize(axis_vector)
    start = origin - axis_vector * scale
    end = origin + axis_vector * scale
    return np.vstack((start, end))


def format_vector_text(prefix: str, row: Dict[str, float]) -> str:
    return (
        f"{prefix} accel = "
        f"[{row[f'{prefix}_ax_mps2']:6.2f}, {row[f'{prefix}_ay_mps2']:6.2f}, {row[f'{prefix}_az_mps2']:6.2f}] m/s^2\n"
        f"{prefix} gyro  = "
        f"[{row[f'{prefix}_gx_dps']:6.2f}, {row[f'{prefix}_gy_dps']:6.2f}, {row[f'{prefix}_gz_dps']:6.2f}] deg/s"
    )


def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    require_numpy()
    plt, FuncAnimation = require_matplotlib()

    if not args.demo and not args.port:
        print("--port or --demo is required.", file=sys.stderr)
        return 2

    csv_file = None
    writer = None
    csv_path = None
    if args.csv:
        csv_path = resolve_output_path(args.csv, ".csv")
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=VIEWER_FIELDS)
        writer.writeheader()
        csv_file.flush()

    save_figure_path = None
    if args.save_figure:
        save_figure_path = resolve_output_path(args.save_figure, ".png")

    buffer = LiveBuffer(window_seconds=args.window_seconds)
    producer = None
    serial_port = None

    try:
        if args.demo:
            producer = DemoProducer(buffer)
            producer.start()
        else:
            serial_port = open_serial(args.port, args.baud)
            print(f"Connected to {args.port} @ {args.baud}")
            if args.startup_delay > 0.0:
                print(f"Waiting {args.startup_delay:.1f}s for board startup")
                time.sleep(args.startup_delay)
            wait_for_viewer_header(serial_port, timeout_s=args.header_timeout)
            producer = SerialProducer(serial_port, buffer, writer=writer, csv_file=csv_file)
            producer.start()

        figure = plt.figure(figsize=(16, 9), facecolor="#f5f2ea")
        grid = figure.add_gridspec(
            4,
            2,
            width_ratios=[1.7, 1.0],
            height_ratios=[1.0, 1.0, 1.0, 0.9],
            wspace=0.16,
            hspace=0.26,
        )

        ax_pose = figure.add_subplot(grid[:, 0], projection="3d")
        ax_accel = figure.add_subplot(grid[0, 1])
        ax_gyro = figure.add_subplot(grid[1, 1], sharex=ax_accel)
        ax_angle = figure.add_subplot(grid[2, 1], sharex=ax_accel)
        ax_info = figure.add_subplot(grid[3, 1])

        for axis in (ax_accel, ax_gyro, ax_angle):
            axis.set_facecolor("#fffdf8")
            axis.grid(True, alpha=0.18)
        ax_info.set_facecolor("#fffdf8")
        ax_info.axis("off")

        ax_pose.set_facecolor("#fffdf8")
        ax_pose.set_title("3D Live Estimated Pose", fontsize=14, pad=16)
        ax_pose.set_xlabel("X")
        ax_pose.set_ylabel("Y")
        ax_pose.set_zlabel("Z")
        ax_pose.view_init(elev=24, azim=-62)
        try:
            ax_pose.set_box_aspect((1.15, 1.0, 0.92))
        except AttributeError:
            pass

        total_reach = args.base_height + args.upper_length + args.forearm_length
        ax_pose.set_xlim(-0.18 * total_reach, 0.92 * total_reach)
        ax_pose.set_ylim(-0.58 * total_reach, 0.58 * total_reach)
        ax_pose.set_zlim(-0.04 * total_reach, args.base_height + args.upper_length + 0.92 * args.forearm_length)

        reference_points = reference_chain_points(
            args.base_height,
            args.upper_length,
            args.forearm_length,
        )

        ax_pose.plot(
            reference_points[:, 0],
            reference_points[:, 1],
            reference_points[:, 2],
            linestyle="--",
            linewidth=2.6,
            color="#a0a6b8",
            alpha=0.75,
            label="reference pose",
        )
        actual_line, = ax_pose.plot(
            [],
            [],
            [],
            "o-",
            linewidth=4.6,
            markersize=8.5,
            color="#1f77b4",
            label="live estimated pose",
        )
        wrist_trace_line, = ax_pose.plot(
            [],
            [],
            [],
            linewidth=2.2,
            color="#d1495b",
            alpha=0.85,
            label="tool trace",
        )
        upper_axis_line, = ax_pose.plot(
            [],
            [],
            [],
            linestyle="-.",
            linewidth=2.0,
            color="#e08e0b",
            alpha=0.95,
            label="hinge axis",
        )
        forearm_axis_line, = ax_pose.plot(
            [],
            [],
            [],
            linestyle="-.",
            linewidth=2.0,
            color="#2a9d8f",
            alpha=0.95,
        )
        pose_status_text = ax_pose.text2D(0.03, 0.95, "", transform=ax_pose.transAxes, va="top")
        ax_pose.legend(loc="upper right")

        accel_lines = {
            "x": ax_accel.plot([], [], color="#d1495b", linewidth=1.7, label="ax")[0],
            "y": ax_accel.plot([], [], color="#edae49", linewidth=1.7, label="ay")[0],
            "z": ax_accel.plot([], [], color="#00798c", linewidth=1.9, label="az")[0],
        }
        gyro_lines = {
            "x": ax_gyro.plot([], [], color="#30638e", linewidth=1.7, label="gx")[0],
            "y": ax_gyro.plot([], [], color="#8e5572", linewidth=1.7, label="gy")[0],
            "z": ax_gyro.plot([], [], color="#3b7a57", linewidth=1.9, label="gz")[0],
        }
        angle_lines = {
            "primary": ax_angle.plot([], [], color="#d1495b", linewidth=1.7, linestyle="--", label="elbow primary")[0],
            "backup": ax_angle.plot([], [], color="#7b2cbf", linewidth=1.6, linestyle=":", label="elbow backup")[0],
            "fused": ax_angle.plot([], [], color="#1f77b4", linewidth=2.6, label="elbow fused")[0],
            "upper": ax_angle.plot([], [], color="#f18f43", linewidth=1.7, label="upper tilt")[0],
            "forearm": ax_angle.plot([], [], color="#2a9d8f", linewidth=1.7, label="forearm tilt")[0],
        }

        ax_accel.set_title("Selected IMU Accel", fontsize=12)
        ax_gyro.set_title("Selected IMU Gyro", fontsize=12)
        ax_angle.set_title("Estimated Angles", fontsize=12)
        ax_accel.set_ylabel("m/s^2")
        ax_gyro.set_ylabel("deg/s")
        ax_angle.set_ylabel("deg")
        ax_angle.set_xlabel("Recent Time Window (s)")

        for axis in (ax_accel, ax_gyro, ax_angle):
            axis.set_xlim(0.0, args.window_seconds)
            axis.legend(loc="upper left", ncol=2, fontsize=9)

        info_text = ax_info.text(
            0.02,
            0.98,
            "Waiting for data",
            va="top",
            ha="left",
            fontsize=10.2,
            family="monospace",
        )

        selected_sensor_key = "1"

        def on_key_press(event) -> None:
            nonlocal selected_sensor_key
            if event.key in SENSOR_OPTIONS:
                selected_sensor_key = event.key
                return

            if isinstance(producer, SerialProducer) and event.key in {"r", "e", "g", "x", "p", "v", "h"}:
                producer.send(event.key)

        figure.canvas.mpl_connect("key_press_event", on_key_press)

        def update(_frame_index):
            rows, sample_count, last_message, first_timestamp_s = buffer.snapshot()
            artists = [actual_line, wrist_trace_line, upper_axis_line, forearm_axis_line, pose_status_text, info_text]
            artists.extend(accel_lines.values())
            artists.extend(gyro_lines.values())
            artists.extend(angle_lines.values())

            if not rows:
                pose_status_text.set_text("Waiting for sensor data")
                info_text.set_text(
                    "Waiting for data\n"
                    "\n"
                    "Keys\n"
                    "1/2/3 : select displayed IMU\n"
                    "r/e/g/x/p/v/h : send board command\n"
                )
                return artists

            latest = rows[-1]
            latest_time_s = latest["millis"] / 1000.0
            start_time_s = first_timestamp_s if first_timestamp_s is not None else latest_time_s
            duration_s = max(0.0, latest_time_s - start_time_s)
            visible_start_s = max(0.0, latest_time_s - args.window_seconds)
            x_values = [(row["millis"] / 1000.0) - visible_start_s for row in rows]

            points = build_chain_points(
                latest,
                base_height=args.base_height,
                upper_length=args.upper_length,
                forearm_length=args.forearm_length,
            )
            actual_line.set_data(points[:, 0], points[:, 1])
            actual_line.set_3d_properties(points[:, 2])

            trace_points = np.array(
                [
                    build_chain_points(
                        row,
                        base_height=args.base_height,
                        upper_length=args.upper_length,
                        forearm_length=args.forearm_length,
                    )[-1]
                    for row in rows
                ],
                dtype=float,
            )
            wrist_trace_line.set_data(trace_points[:, 0], trace_points[:, 1])
            wrist_trace_line.set_3d_properties(trace_points[:, 2])

            upper_axis_segment = axis_segment(
                points[1],
                np.array(
                    [latest["upper_hinge_x"], latest["upper_hinge_y"], latest["upper_hinge_z"]],
                    dtype=float,
                ),
                scale=0.07,
            )
            forearm_axis_segment = axis_segment(
                points[2],
                np.array(
                    [latest["forearm_hinge_x"], latest["forearm_hinge_y"], latest["forearm_hinge_z"]],
                    dtype=float,
                ),
                scale=0.07,
            )
            upper_axis_line.set_data(upper_axis_segment[:, 0], upper_axis_segment[:, 1])
            upper_axis_line.set_3d_properties(upper_axis_segment[:, 2])
            forearm_axis_line.set_data(forearm_axis_segment[:, 0], forearm_axis_segment[:, 1])
            forearm_axis_line.set_3d_properties(forearm_axis_segment[:, 2])

            sensor_prefix, sensor_label = SENSOR_OPTIONS[selected_sensor_key]
            ax_accel.set_title(f"Selected IMU Accel: {sensor_label}", fontsize=12)
            ax_gyro.set_title(f"Selected IMU Gyro: {sensor_label}", fontsize=12)

            accel_series = {
                "x": [row[f"{sensor_prefix}_ax_mps2"] for row in rows],
                "y": [row[f"{sensor_prefix}_ay_mps2"] for row in rows],
                "z": [row[f"{sensor_prefix}_az_mps2"] for row in rows],
            }
            gyro_series = {
                "x": [row[f"{sensor_prefix}_gx_dps"] for row in rows],
                "y": [row[f"{sensor_prefix}_gy_dps"] for row in rows],
                "z": [row[f"{sensor_prefix}_gz_dps"] for row in rows],
            }
            angle_series = {
                "primary": [row["elbow_primary_deg"] for row in rows],
                "backup": [row["elbow_backup_deg"] for row in rows],
                "fused": [row["elbow_fused_deg"] for row in rows],
                "upper": [row["upper_tilt_deg"] for row in rows],
                "forearm": [row["forearm_tilt_deg"] for row in rows],
            }

            for axis_key, line in accel_lines.items():
                line.set_data(x_values, accel_series[axis_key])
            for axis_key, line in gyro_lines.items():
                line.set_data(x_values, gyro_series[axis_key])
            for line_key, line in angle_lines.items():
                line.set_data(x_values, angle_series[line_key])

            for axis in (ax_accel, ax_gyro, ax_angle):
                axis.set_xlim(0.0, args.window_seconds)

            autoscale_axis(ax_accel, list(accel_series.values()), default_half_range=12.0)
            autoscale_axis(ax_gyro, list(gyro_series.values()), default_half_range=110.0)
            autoscale_axis(ax_angle, list(angle_series.values()), default_half_range=90.0)

            mode_name = MODE_NAMES.get(int(round(latest["mode"])), f"mode-{int(round(latest['mode']))}")
            pose_status_text.set_text(
                f"samples {sample_count} | elapsed {duration_s:6.2f}s\n"
                f"mode = {mode_name} | estimate_valid = {int(round(latest['estimate_valid']))}\n"
                f"elbow = {latest['elbow_fused_deg']:7.2f} deg | upper = {latest['upper_tilt_deg']:7.2f} deg | forearm = {latest['forearm_tilt_deg']:7.2f} deg"
            )

            info_text.set_text(
                f"stream: {'DEMO' if args.demo else args.port}\n"
                f"selected IMU: {sensor_label}\n"
                f"samples: {sample_count}\n"
                f"last board message: {last_message}\n"
                f"\n"
                f"elbow primary = {latest['elbow_primary_deg']:7.2f} deg\n"
                f"elbow backup  = {latest['elbow_backup_deg']:7.2f} deg\n"
                f"elbow fused   = {latest['elbow_fused_deg']:7.2f} deg\n"
                f"upper tilt    = {latest['upper_tilt_deg']:7.2f} deg\n"
                f"forearm tilt  = {latest['forearm_tilt_deg']:7.2f} deg\n"
                f"upper consistency = {latest['upper_consistency_deg']:6.2f} deg\n"
                f"forearm penalty   = {latest['forearm_dynamic_penalty_mps2']:6.2f} m/s^2\n"
                f"upper conf        = {latest['upper_confidence']:6.3f}\n"
                f"forearm conf      = {latest['forearm_confidence']:6.3f}\n"
                f"\n"
                f"{format_vector_text('u1', latest)}\n"
                f"\n"
                f"{format_vector_text('u2', latest)}\n"
                f"\n"
                f"{format_vector_text('f1', latest)}\n"
                f"\n"
                f"Keys\n"
                f"1/2/3 : switch displayed IMU\n"
                f"r : reference calibration\n"
                f"e : elbow-axis calibration\n"
                f"g : start continuous estimation\n"
                f"x : stop\n"
                f"p : print header\n"
                f"v : viewer CSV on/off\n"
            )

            return artists

        def handle_close(_event) -> None:
            if producer is not None:
                producer.stop()

        figure.canvas.mpl_connect("close_event", handle_close)
        figure.suptitle("AlphaF Unit 1 IMU Live Validation", fontsize=16, y=0.985)
        animation = FuncAnimation(
            figure,
            update,
            interval=args.refresh_ms,
            cache_frame_data=False,
            blit=False,
        )
        figure._viewer_animation = animation
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        plt.show()

        if save_figure_path is not None:
            update(0)
            figure.savefig(save_figure_path, dpi=150)
            print(f"Saved figure to: {save_figure_path}")

        if isinstance(producer, SerialProducer) and producer.error is not None:
            print(producer.error, file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("Viewer interrupted by user.")
    except TimeoutError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        if producer is not None:
            producer.stop()
            producer.join(timeout=2.0)
        if serial_port is not None:
            serial_port.close()
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
