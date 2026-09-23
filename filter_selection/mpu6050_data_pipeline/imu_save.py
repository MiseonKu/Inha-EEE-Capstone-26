import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

from imu_processing import CSV_FIELDS as OUTPUT_CSV_FIELDS
from imu_processing import measurement_row_to_csv_row, serial_row_to_measurement_row


SERIAL_FIELDS = [
    "millis",
    "ax_raw",
    "ay_raw",
    "az_raw",
    "gx_raw",
    "gy_raw",
    "gz_raw",
    "temp_raw",
    "temp_c",
]
LEGACY_SERIAL_FIELDS = SERIAL_FIELDS[:-1]
PREFIXED_HEADER = ["I", *SERIAL_FIELDS]
LEGACY_PREFIXED_HEADER = ["I", *LEGACY_SERIAL_FIELDS]

# Backward-compatible export name used by local scripts.
CSV_FIELDS = OUTPUT_CSV_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture MPU6050 serial data from imu_raw.ino or imu_calib.ino and save engineering-unit CSV into filter_results/experiment_data."
    )
    parser.add_argument("--port", help="Serial port such as COM14")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument(
        "--csv",
        help="CSV output file name. Relative names are saved inside filter_results/experiment_data",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional capture duration in seconds. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after opening the serial port before sending commands",
    )
    parser.add_argument(
        "--header-timeout",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for the IMU CSV header before aborting",
    )
    parser.add_argument(
        "--send-init",
        action="store_true",
        help="Send INIT once after opening the port",
    )
    parser.add_argument(
        "--no-send-imu",
        action="store_true",
        help="Do not send IMU while waiting for the CSV header",
    )
    return parser.parse_args()


def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        description = port.description or "Unknown device"
        print(f"{port.device}: {description}")


def resolve_csv_path(csv_arg: str | None) -> Path:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir.parent / "filter_results" / "experiment_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_arg:
        timestamp = datetime.now().strftime("imu_save_%Y%m%d_%H%M%S.csv")
        return output_dir / timestamp

    candidate = Path(csv_arg)
    if candidate.is_absolute():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    return output_dir / candidate


def open_serial(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=1.0,
        rtscts=False,
        dsrdtr=False,
    )

    try:
        ser.setDTR(False)
        ser.setRTS(False)
    except OSError:
        pass

    return ser


def send_command(ser: serial.Serial, command: str) -> None:
    ser.write((command.strip() + "\n").encode("utf-8"))
    ser.flush()


def classify_header(fields: list[str]) -> str | None:
    if fields == SERIAL_FIELDS:
        return "plain"
    if fields == LEGACY_SERIAL_FIELDS:
        return "plain-legacy"
    if fields == PREFIXED_HEADER:
        return "prefixed"
    if fields == LEGACY_PREFIXED_HEADER:
        return "prefixed-legacy"
    return None


def wait_for_header(ser: serial.Serial, send_imu: bool, timeout_s: float) -> str:
    print("Waiting for IMU CSV header from board...")
    wait_start = time.monotonic()
    last_imu_send = 0.0
    imu_announced = False

    while True:
        now = time.monotonic()
        if timeout_s > 0 and now - wait_start >= timeout_s:
            raise TimeoutError(
                "Timed out waiting for IMU CSV header. Upload imu_raw.ino, imu_calib.ino, "
                "or another sketch that streams millis,ax_raw,...,temp_raw[,temp_c] and try again."
            )

        if send_imu and now - last_imu_send >= 1.0:
            if not imu_announced:
                print("Sending IMU command...")
                imu_announced = True
            send_command(ser, "IMU")
            last_imu_send = now

        raw = ser.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if line.startswith("#"):
            print(line)
            continue

        fields = [part.strip() for part in line.split(",")]
        header_mode = classify_header(fields)
        if header_mode:
            print(f"Header received ({header_mode}).")
            return header_mode

        data_row = parse_data_line(line)
        if data_row is not None:
            print("Data stream detected without a fresh header. Continuing...")
            return "data-only"

        print(f"Skipping non-data line: {line}")


def parse_data_line(line: str) -> dict[str, str] | None:
    fields = [part.strip() for part in line.split(",")]
    if classify_header(fields):
        return None

    if len(fields) == len(PREFIXED_HEADER) and fields[0] == "I":
        values = fields[1:]
        field_names = SERIAL_FIELDS
    elif len(fields) == len(SERIAL_FIELDS):
        values = fields
        field_names = SERIAL_FIELDS
    elif len(fields) == len(LEGACY_PREFIXED_HEADER) and fields[0] == "I":
        values = fields[1:]
        field_names = LEGACY_SERIAL_FIELDS
    elif len(fields) == len(LEGACY_SERIAL_FIELDS):
        values = fields
        field_names = LEGACY_SERIAL_FIELDS
    else:
        return None

    row = {field: "" for field in SERIAL_FIELDS}
    try:
        for key, value in zip(field_names, values):
            if key == "temp_c":
                float(value)
            else:
                int(value)
            row[key] = value
    except ValueError:
        return None

    if not row["temp_c"] and row["temp_raw"]:
        row["temp_c"] = f"{(int(row['temp_raw']) / 340.0) + 36.53:.2f}"

    return row


def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    if not args.port:
        print("--port is required unless --list-ports is used.", file=sys.stderr)
        return 2

    csv_path = resolve_csv_path(args.csv)
    sample_count = 0

    try:
        with open_serial(args.port, args.baud) as ser, csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            csv_file.flush()

            print(f"Connected to {args.port} @ {args.baud}")
            print(f"Saving CSV to {csv_path}")

            if args.startup_delay > 0:
                print(f"Waiting {args.startup_delay:.1f}s for board startup...")
                time.sleep(args.startup_delay)

            if args.send_init:
                print("Sending INIT command...")
                send_command(ser, "INIT")

            wait_for_header(
                ser,
                send_imu=not args.no_send_imu,
                timeout_s=args.header_timeout,
            )

            capture_start = time.monotonic()
            while True:
                if args.duration > 0 and time.monotonic() - capture_start >= args.duration:
                    print("Capture duration reached.")
                    break

                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if line.startswith("#"):
                    print(line)
                    continue

                row = parse_data_line(line)
                if row is None:
                    print(f"Skipping malformed line: {line}")
                    continue

                measurement_row = serial_row_to_measurement_row(row)
                writer.writerow(measurement_row_to_csv_row(measurement_row))
                csv_file.flush()
                sample_count += 1

                if sample_count % 25 == 0:
                    print(f"Saved {sample_count} samples...")
    except KeyboardInterrupt:
        print("Capture interrupted by user.")
    except TimeoutError as exc:
        print(exc, file=sys.stderr)
        return 1

    if sample_count == 0:
        print(
            "Finished with 0 samples saved. The board opened, but no IMU data rows arrived. "
            "Check the Arduino serial output for MPU6050 detection, wiring, and sensor model.",
            file=sys.stderr,
        )
        return 1

    print(f"Finished with {sample_count} samples saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
