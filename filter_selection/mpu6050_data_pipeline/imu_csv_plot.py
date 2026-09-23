from __future__ import annotations

import argparse
import sys

from imu_processing import (
    ACCEL_UNIT_FIELDS,
    DEFAULT_EMA_ALPHA,
    EmaSmoother,
    GYRO_UNIT_FIELDS,
    PLOT_FIELDS,
    raw_row_for_plot,
    read_csv_rows,
    resolve_input_csv_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a saved IMU CSV with smoothing."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file. Relative names are searched in filter_results/experiment_data first.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help="EMA smoothing factor. Use 1.0 for no smoothing.",
    )
    parser.add_argument(
        "--save-figure",
        help="Optional image path to save the plotted figure instead of only showing it.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()

    input_path = resolve_input_csv_path(args.input)
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required. Install it with: pip install matplotlib", file=sys.stderr)
        return 1

    rows = read_csv_rows(input_path)
    if not rows:
        print(f"CSV has no rows: {input_path}", file=sys.stderr)
        return 1

    title = "MPU6050 CSV Plot"
    transform_row = raw_row_for_plot

    smoother = EmaSmoother(fields=PLOT_FIELDS, alpha=args.alpha)
    processed_rows = [smoother.apply(transform_row(row)) for row in rows]

    timestamps = [row["millis"] / 1000.0 for row in processed_rows]
    time_zero = timestamps[0]
    x_values = [timestamp - time_zero for timestamp in timestamps]

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    figure.suptitle(title)

    for field in ACCEL_UNIT_FIELDS:
        axes[0].plot(x_values, [row[field] for row in processed_rows], label=field)
    for field in GYRO_UNIT_FIELDS:
        axes[1].plot(x_values, [row[field] for row in processed_rows], label=field)
    axes[2].plot(x_values, [row["temp_c"] for row in processed_rows], label="temp_c", color="#7f5539")

    axes[0].set_ylabel("Accel (m/s^2)")
    axes[1].set_ylabel("Gyro (deg/s)")
    axes[2].set_ylabel("Temp C")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    autoscale_axis(axes[0], [[row[field] for row in processed_rows] for field in ACCEL_UNIT_FIELDS])
    autoscale_axis(axes[1], [[row[field] for row in processed_rows] for field in GYRO_UNIT_FIELDS])
    autoscale_axis(axes[2], [[row["temp_c"] for row in processed_rows]])

    figure.tight_layout(rect=(0, 0, 1, 0.95))

    if args.save_figure:
        figure.savefig(args.save_figure, dpi=150)
        print(f"Saved figure to {args.save_figure}")

    if "agg" in plt.get_backend().lower() and args.save_figure:
        return 0

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
