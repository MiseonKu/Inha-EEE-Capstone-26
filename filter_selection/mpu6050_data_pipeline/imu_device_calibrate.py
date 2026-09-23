from __future__ import annotations

import argparse
import sys
import time

from imu_save import list_ports, open_serial, send_command, wait_for_header


ACTION_TO_COMMAND = {
    "calibrate": "CALIBRATE",
    "show": "SHOW",
    "clear": "CLEAR",
    "help": "HELP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send calibration commands to imu_calib.ino and print the board response."
    )
    parser.add_argument("--port", help="Serial port such as COM14")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")
    parser.add_argument(
        "--action",
        choices=sorted(ACTION_TO_COMMAND.keys()),
        default="calibrate",
        help="Board command to send",
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
        help="Maximum seconds to wait for the board CSV header before sending the command",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for the command to complete",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    if not args.port:
        print("--port is required unless --list-ports is used.", file=sys.stderr)
        return 2

    command = ACTION_TO_COMMAND[args.action]
    completion_markers = {
        "calibrate": ["# Calibration complete and saved to EEPROM."],
        "clear": ["# Calibration cleared from EEPROM."],
        "show": ["# accel_offset_raw=", "# Calibration: not set"],
        "help": ["# Commands:"],
    }

    try:
        with open_serial(args.port, args.baud) as ser:
            print(f"Connected to {args.port} @ {args.baud}")

            if args.startup_delay > 0:
                print(f"Waiting {args.startup_delay:.1f}s for board startup...")
                time.sleep(args.startup_delay)

            wait_for_header(ser, send_imu=False, timeout_s=args.header_timeout)

            print(f"Sending command: {command}")
            send_command(ser, command)

            deadline = time.monotonic() + max(args.timeout, 1.0)
            lines_seen = 0

            while time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                print(line)
                lines_seen += 1

                if any(marker in line for marker in completion_markers[args.action]):
                    if args.action == "show":
                        return 0
                    if args.action == "help":
                        return 0
                    if args.action == "clear":
                        return 0
                    if args.action == "calibrate":
                        return 0

                if args.action in {"show", "help"} and lines_seen >= 4:
                    return 0

            print(f"Timed out waiting for {args.action} response.", file=sys.stderr)
            return 1
    except TimeoutError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
