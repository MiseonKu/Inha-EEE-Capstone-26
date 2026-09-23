from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from pathlib import Path

from imu_processing import (
    ACCEL_UNIT_FIELDS,
    CSV_FIELDS,
    DEFAULT_EMA_ALPHA,
    EmaSmoother,
    GYRO_UNIT_FIELDS,
    PLOT_FIELDS,
    measurement_row_to_csv_row,
    raw_row_for_plot,
)
from imu_save import list_ports, open_serial, parse_data_line, resolve_csv_path, wait_for_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot live MPU6050 waveforms streamed by imu_raw.ino or imu_calib.ino."
    )
    parser.add_argument("--port", help="Serial port such as COM14")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Visible time window in seconds",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=100,
        help="Plot refresh interval in milliseconds",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after opening the serial port",
    )
    parser.add_argument(
        "--header-timeout",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for the IMU CSV header",
    )
    parser.add_argument(
        "--csv",
        help="Optional engineering-unit CSV output file name. Relative names are saved inside filter_results/experiment_data",
    )
    parser.add_argument(
        "--save-figure",
        help="Optional waveform image file name. Relative names are saved inside filter_results/experiment_data",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="EMA smoothing factor. Use 1.0 for no smoothing.",
    )
    return parser.parse_args()


class LiveBuffer:
    def __init__(self, *, window_seconds: float, fields: tuple[str, ...]) -> None:
        self.window_seconds = max(window_seconds, 1.0)
        self.fields = fields
        self.timestamps = deque()
        self.values = {field: deque() for field in fields}
        self.sample_count = 0
        self.last_message = "Waiting for IMU data..."
        self._lock = threading.Lock()

    def add_row(self, row: dict[str, float]) -> None:
        timestamp_s = row["millis"] / 1000.0

        with self._lock:
            self.timestamps.append(timestamp_s)
            for field in self.fields:
                self.values[field].append(float(row[field]))
            self.sample_count += 1
            self.last_message = f"Samples: {self.sample_count}"
            self._trim_locked()

    def set_message(self, message: str) -> None:
        with self._lock:
            self.last_message = message

    def snapshot(self) -> tuple[list[float], dict[str, list[float]], int, str]:
        with self._lock:
            return (
                list(self.timestamps),
                {field: list(series) for field, series in self.values.items()},
                self.sample_count,
                self.last_message,
            )

    def _trim_locked(self) -> None:
        if not self.timestamps:
            return

        cutoff = self.timestamps[-1] - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            for field in self.fields:
                self.values[field].popleft()


class SerialReader(threading.Thread):
    def __init__(
        self,
        *,
        serial_port,
        buffer: LiveBuffer,
        transform_row,
        smoother: EmaSmoother,
        writer: csv.DictWriter | None = None,
        csv_file=None,
    ) -> None:
        super().__init__(daemon=True)
        self.serial_port = serial_port
        self.buffer = buffer
        self.transform_row = transform_row
        self.smoother = smoother
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

                transformed = self.transform_row(parsed)
                self.buffer.add_row(self.smoother.apply(transformed))

                if self.writer is not None:
                    self.writer.writerow(measurement_row_to_csv_row(transformed))
                    pending_rows += 1
                    if pending_rows >= flush_every and self.csv_file is not None:
                        self.csv_file.flush()
                        pending_rows = 0
        except Exception as exc:
            self.error = f"Serial reader stopped: {exc}"
        finally:
            if self.csv_file is not None:
                self.csv_file.flush()


def autoscale_axis(ax, series_list: list[list[float]]) -> None:
    values = [value for series in series_list for value in series]
    if not values:
        return

    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.1, 1.0)
    else:
        pad = max((high - low) * 0.1, 1.0 if abs(high) > 1.0 else 0.05)

    ax.set_ylim(low - pad, high + pad)


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


def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    if not args.port:
        print("--port is required unless --list-ports is used.", file=sys.stderr)
        return 2

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib is required. Install it with: pip install matplotlib", file=sys.stderr)
        return 1

    title = "MPU6050 Live Waveforms"
    transform_row = raw_row_for_plot

    smoother = EmaSmoother(fields=PLOT_FIELDS, alpha=args.alpha)
    buffer = LiveBuffer(window_seconds=args.window_seconds, fields=PLOT_FIELDS)

    csv_file = None
    writer = None
    csv_path: Path | None = None
    figure_path: Path | None = None

    if args.csv:
        csv_path = resolve_csv_path(args.csv)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

    if args.save_figure:
        figure_path = resolve_figure_path(args.save_figure)

    try:
        with open_serial(args.port, args.baud) as ser:
            print(f"Connected to {args.port} @ {args.baud}")
            if csv_path is not None:
                print(f"Saving engineering-unit CSV to {csv_path}")
            if figure_path is not None:
                print(f"Waveform image will be saved to {figure_path} when the window closes")

            if args.startup_delay > 0:
                print(f"Waiting {args.startup_delay:.1f}s for board startup...")
                time.sleep(args.startup_delay)

            wait_for_header(ser, send_imu=False, timeout_s=args.header_timeout)

            reader = SerialReader(
                serial_port=ser,
                buffer=buffer,
                transform_row=transform_row,
                smoother=smoother,
                writer=writer,
                csv_file=csv_file,
            )
            reader.start()

            figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            figure.suptitle(title)

            accel_lines = {
                ACCEL_UNIT_FIELDS[0]: axes[0].plot([], [], label=ACCEL_UNIT_FIELDS[0], color="#d1495b")[0],
                ACCEL_UNIT_FIELDS[1]: axes[0].plot([], [], label=ACCEL_UNIT_FIELDS[1], color="#edae49")[0],
                ACCEL_UNIT_FIELDS[2]: axes[0].plot([], [], label=ACCEL_UNIT_FIELDS[2], color="#00798c")[0],
            }
            gyro_lines = {
                GYRO_UNIT_FIELDS[0]: axes[1].plot([], [], label=GYRO_UNIT_FIELDS[0], color="#30638e")[0],
                GYRO_UNIT_FIELDS[1]: axes[1].plot([], [], label=GYRO_UNIT_FIELDS[1], color="#8e5572")[0],
                GYRO_UNIT_FIELDS[2]: axes[1].plot([], [], label=GYRO_UNIT_FIELDS[2], color="#3b7a57")[0],
            }
            temp_line = axes[2].plot([], [], label="temp_c", color="#7f5539")[0]

            axes[0].set_ylabel("Accel (m/s^2)")
            axes[1].set_ylabel("Gyro (deg/s)")
            axes[2].set_ylabel("Temp C")
            axes[2].set_xlabel("Time in Window (s)")

            for ax in axes:
                ax.grid(True, alpha=0.15)
                ax.legend(loc="upper left")
                ax.set_xlim(0, args.window_seconds)

            status_text = figure.text(0.01, 0.98, "Waiting for IMU data...", va="top")

            def update(_frame):
                timestamps, values, sample_count, last_message = buffer.snapshot()
                artists = [status_text, temp_line]
                artists.extend(accel_lines.values())
                artists.extend(gyro_lines.values())

                if not timestamps:
                    status_text.set_text(last_message)
                    return artists

                newest_time = timestamps[-1]
                window_start = max(0.0, newest_time - args.window_seconds)
                x_values = [timestamp - window_start for timestamp in timestamps]

                for field, line in accel_lines.items():
                    line.set_data(x_values, values[field])
                for field, line in gyro_lines.items():
                    line.set_data(x_values, values[field])
                temp_line.set_data(x_values, values["temp_c"])

                for ax in axes:
                    ax.set_xlim(0, args.window_seconds)

                autoscale_axis(axes[0], [values[field] for field in ACCEL_UNIT_FIELDS])
                autoscale_axis(axes[1], [values[field] for field in GYRO_UNIT_FIELDS])
                autoscale_axis(axes[2], [values["temp_c"]])

                status_text.set_text(
                    f"Samples: {sample_count} | Latest board time: {newest_time:.2f}s | {last_message}"
                )
                return artists

            def handle_close(_event) -> None:
                reader.stop()

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

            if figure_path is not None:
                update(None)
                figure.savefig(figure_path, dpi=150)
                print(f"Saved waveform image to {figure_path}")

            if reader.error is not None:
                print(reader.error, file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        print("Plotting interrupted by user.")
    finally:
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
