from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev


FILTERS = ("none", "complementary", "kalman", "mahony", "madgwick")
AXES = ("roll", "pitch", "yaw")
DATA_FOLDERS = (
    "none_straight",
    "complementary_straight",
    "kalman_straight",
    "mahony_straight",
    "madgwick_straight",
)
FILTER_LABELS = {
    "none": "None",
    "complementary": "Complementary",
    "kalman": "Kalman",
    "mahony": "Mahony",
    "madgwick": "Madgwick",
}
FILTER_COLORS = {
    "none": "#8A5A3B",
    "complementary": "#D1495B",
    "kalman": "#0F7C8A",
    "mahony": "#3F6F9D",
    "madgwick": "#4C8B63",
}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir.parent / "filter_results"
    parser = argparse.ArgumentParser(
        description="Build paper-friendly stability figures from straight IMU trials."
    )
    parser.add_argument(
        "--input-root",
        default=str(results_dir / "experiment_data"),
        help="Directory containing the straight trial folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(results_dir),
        help="Directory where the figure and summary CSV will be saved.",
    )
    return parser.parse_args()


def iter_input_csv_paths(input_root: Path) -> list[Path]:
    csv_paths: list[Path] = []
    for folder_name in DATA_FOLDERS:
        folder_path = input_root / folder_name
        csv_paths.extend(sorted(folder_path.glob("imu_relative_pose_*.csv")))
    return csv_paths


def load_tracking_series(csv_path: Path) -> dict[str, dict[str, list[float]]]:
    series = {
        filter_name: {axis: [] for axis in AXES}
        for filter_name in FILTERS
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "tracking":
                continue

            for filter_name in FILTERS:
                values: list[float] = []
                for axis in AXES:
                    raw = row.get(f"{filter_name}_{axis}_deg", "")
                    if raw == "":
                        values = []
                        break
                    values.append(float(raw))

                if len(values) != len(AXES):
                    continue

                for axis, value in zip(AXES, values):
                    series[filter_name][axis].append(value)

    return series


def compute_trial_metrics(series: dict[str, list[float]]) -> dict[str, float] | None:
    if any(not series[axis] for axis in AXES):
        return None

    axis_std = {axis: pstdev(series[axis]) for axis in AXES}
    axis_mean_abs = {
        axis: mean(abs(value) for value in series[axis])
        for axis in AXES
    }
    axis_final_abs = {
        axis: abs(series[axis][-1])
        for axis in AXES
    }

    composite_score = sum(
        axis_std[axis] + axis_mean_abs[axis] + axis_final_abs[axis]
        for axis in AXES
    )
    final_drift_norm = math.sqrt(
        sum(series[axis][-1] ** 2 for axis in AXES)
    )
    mean_jitter = mean(axis_std.values())
    mean_abs_angle = mean(axis_mean_abs.values())

    return {
        "mean_jitter": mean_jitter,
        "final_drift_norm": final_drift_norm,
        "mean_abs_angle": mean_abs_angle,
        "composite_score": composite_score,
    }

def build_trial_metrics(input_root: Path) -> dict[str, list[dict[str, float]]]:
    metrics_by_filter = {filter_name: [] for filter_name in FILTERS}

    for csv_path in iter_input_csv_paths(input_root):
        trial_series = load_tracking_series(csv_path)
        for filter_name in FILTERS:
            metrics = compute_trial_metrics(trial_series[filter_name])
            if metrics is not None:
                metrics_by_filter[filter_name].append(metrics)

    return metrics_by_filter


def build_summary_rows(
    metrics_by_filter: dict[str, list[dict[str, float]]]
) -> list[dict[str, str]]:
    composite_mean_by_filter = {
        filter_name: mean(
            metric["composite_score"] for metric in metrics_by_filter[filter_name]
        )
        for filter_name in FILTERS
    }
    composite_median_by_filter = {
        filter_name: median(
            metric["composite_score"] for metric in metrics_by_filter[filter_name]
        )
        for filter_name in FILTERS
    }
    none_composite_mean = composite_mean_by_filter["none"]
    filter_order = sorted(
        FILTERS,
        key=lambda name: composite_mean_by_filter[name],
    )
    rank_lookup = {
        filter_name: rank
        for rank, filter_name in enumerate(filter_order, start=1)
    }

    summary_rows: list[dict[str, str]] = []
    for filter_name in filter_order:
        metrics = metrics_by_filter[filter_name]
        jitter_values = [metric["mean_jitter"] for metric in metrics]
        drift_values = [metric["final_drift_norm"] for metric in metrics]
        abs_angle_values = [metric["mean_abs_angle"] for metric in metrics]
        summary_rows.append(
            {
                "rank": str(rank_lookup[filter_name]),
                "filter": filter_name,
                "trial_count": str(len(metrics)),
                "raw_composite_score_mean": f"{composite_mean_by_filter[filter_name]:.3f}",
                "raw_composite_score_median": f"{composite_median_by_filter[filter_name]:.3f}",
                "improvement_vs_none_pct": (
                    f"{(1.0 - (composite_mean_by_filter[filter_name] / none_composite_mean)) * 100.0:.3f}"
                ),
                "jitter_mean_deg": f"{mean(jitter_values):.3f}",
                "jitter_std_deg": f"{pstdev(jitter_values):.3f}",
                "jitter_median_deg": f"{median(jitter_values):.3f}",
                "endpoint_drift_mean_deg": f"{mean(drift_values):.3f}",
                "endpoint_drift_std_deg": f"{pstdev(drift_values):.3f}",
                "endpoint_drift_median_deg": f"{median(drift_values):.3f}",
                "mean_abs_angle_mean_deg": f"{mean(abs_angle_values):.3f}",
                "mean_abs_angle_std_deg": f"{pstdev(abs_angle_values):.3f}",
            }
        )

    return summary_rows


def save_summary_csv(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "rank",
        "filter",
        "trial_count",
        "raw_composite_score_mean",
        "raw_composite_score_median",
        "improvement_vs_none_pct",
        "jitter_mean_deg",
        "jitter_std_deg",
        "jitter_median_deg",
        "endpoint_drift_mean_deg",
        "endpoint_drift_std_deg",
        "endpoint_drift_median_deg",
        "mean_abs_angle_mean_deg",
        "mean_abs_angle_std_deg",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def create_figure(
    metrics_by_filter: dict[str, list[dict[str, float]]],
    summary_rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    ordered_filters = [row["filter"] for row in summary_rows]
    labels = [FILTER_LABELS[name] for name in ordered_filters]
    colors = [FILTER_COLORS[name] for name in ordered_filters]
    x_positions = list(range(len(labels)))
    x_tick_labels = labels
    jitter_distributions = [
        [metric["mean_jitter"] for metric in metrics_by_filter[name]]
        for name in ordered_filters
    ]
    drift_distributions = [
        [metric["final_drift_norm"] for metric in metrics_by_filter[name]]
        for name in ordered_filters
    ]
    jitter_means = [float(row["jitter_mean_deg"]) for row in summary_rows]
    drift_means = [float(row["endpoint_drift_mean_deg"]) for row in summary_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 11,
        }
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    figure.patch.set_facecolor("#f6f3ee")

    figure.suptitle(
        "Straight-Tracking Stability Comparison Across Attitude Filters",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.95,
        "Raw metrics for the tracking interval. Lower values indicate better stability.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#5a544d",
    )

    jitter_box = axes[0].boxplot(
        jitter_distributions,
        positions=x_positions,
        vert=True,
        patch_artist=True,
        widths=0.54,
        showmeans=True,
        showfliers=False,
        whis=(5, 95),
    )
    axes[0].set_title("(a) Across-Trial Jitter")
    axes[0].set_ylabel("Mean angle std. (deg)")
    style_boxplot(jitter_box, colors)
    for x_position, metric_mean in zip(x_positions, jitter_means):
        axes[0].text(
            x_position,
            metric_mean + 0.7,
            f"{metric_mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#2a2723",
        )

    drift_box = axes[1].boxplot(
        drift_distributions,
        positions=x_positions,
        vert=True,
        patch_artist=True,
        widths=0.54,
        showmeans=True,
        showfliers=False,
        whis=(5, 95),
    )
    axes[1].set_title("(b) Endpoint Drift")
    axes[1].set_ylabel("Final attitude norm (deg)")
    style_boxplot(drift_box, colors)
    for x_position, metric_mean in zip(x_positions, drift_means):
        axes[1].text(
            x_position,
            metric_mean + 1.0,
            f"{metric_mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#2a2723",
        )

    for axis in axes:
        axis.set_facecolor("#fcfbf8")
        axis.grid(True, axis="y", alpha=0.28, color="#b8afa3", linewidth=0.8)
        axis.grid(False, axis="x")
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#8b8276")
        axis.spines["bottom"].set_color("#8b8276")

    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xticks(x_positions, x_tick_labels)
    figure.text(
        0.5,
        0.02,
        "Boxes show the interquartile range, the center line is the median, the triangle marks the mean, and whiskers represent the 5th-95th percentile range.",
        ha="center",
        fontsize=9,
        color="#4f4942",
    )
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.92, hspace=0.35)

    png_path = output_dir / "filter_stability_raw_metrics.png"
    svg_path = output_dir / "filter_stability_raw_metrics.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def create_improvement_figure(
    summary_rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    ordered_filters = [row["filter"] for row in summary_rows]
    labels = [FILTER_LABELS[name] for name in ordered_filters]
    colors = [FILTER_COLORS[name] for name in ordered_filters]
    improvement_values = [float(row["improvement_vs_none_pct"]) for row in summary_rows]
    raw_values = [float(row["raw_composite_score_mean"]) for row in summary_rows]
    x_positions = list(range(len(labels)))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
        }
    )

    figure, axis = plt.subplots(figsize=(9.4, 6.2))
    figure.patch.set_facecolor("#f6f3ee")
    axis.set_facecolor("#fcfbf8")

    bars = axis.bar(
        x_positions,
        improvement_values,
        color=colors,
        edgecolor="#2b2621",
        linewidth=1.0,
        width=0.64,
    )
    axis.axhline(0.0, color="#8b8276", linewidth=1.0)
    axis.set_title("Stability Improvement Relative to the Unfiltered Baseline")
    axis.set_ylabel("Improvement vs. None (%)")
    axis.set_xticks(x_positions, labels)
    axis.grid(True, axis="y", alpha=0.28, color="#b8afa3", linewidth=0.8)
    axis.grid(False, axis="x")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8b8276")
    axis.spines["bottom"].set_color("#8b8276")

    upper_limit = max(6.0, max(improvement_values) + 5.0)
    lower_limit = min(-2.0, min(improvement_values) - 2.0)
    axis.set_ylim(lower_limit, upper_limit)

    for bar, improvement_value, raw_value in zip(bars, improvement_values, raw_values):
        text_y = improvement_value + 0.8 if improvement_value >= 0 else improvement_value - 0.8
        va = "bottom" if improvement_value >= 0 else "top"
        raw_x = bar.get_x() + bar.get_width() / 2.0
        raw_y = max(0.65, improvement_value * 0.45) if improvement_value > 0 else 0.75
        raw_color = "#fcfbf8" if improvement_value > 7.0 else "#2b2621"

        if improvement_value <= 1.0:
            text_y = 1.15
            raw_x += 0.06
            raw_y = 0.35
            raw_color = "#2b2621"

        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            f"{improvement_value:.1f}%",
            ha="center",
            va=va,
            fontsize=10.2,
            fontweight="bold",
            color="#1f1b18",
        )
        axis.text(
            raw_x,
            raw_y,
            f"raw {raw_value:.1f}",
            ha="center",
            va="center",
            fontsize=8.8,
            color=raw_color,
        )

    figure.text(
        0.5,
        0.035,
        "Improvement (%) = 100 x [1 - (filter raw composite score / None raw composite score)]",
        ha="center",
        fontsize=9.2,
        color="#4f4942",
    )
    figure.tight_layout(rect=(0.03, 0.07, 0.99, 0.96))

    png_path = output_dir / "filter_stability_improvement_vs_none.png"
    svg_path = output_dir / "filter_stability_improvement_vs_none.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def style_boxplot(boxplot, colors: list[str]) -> None:
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#2b2621")
        patch.set_linewidth(1.0)
    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.5)
    for whisker in boxplot["whiskers"]:
        whisker.set_color("#4d4841")
        whisker.set_linewidth(1.0)
    for cap in boxplot["caps"]:
        cap.set_color("#4d4841")
        cap.set_linewidth(1.0)
    for mean_marker in boxplot["means"]:
        mean_marker.set_markerfacecolor("#111111")
        mean_marker.set_markeredgecolor("#111111")
        mean_marker.set_markersize(5.0)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_filter = build_trial_metrics(input_root)
    if any(not metrics for metrics in metrics_by_filter.values()):
        missing = [name for name, metrics in metrics_by_filter.items() if not metrics]
        raise SystemExit(f"No valid tracking data found for: {', '.join(missing)}")

    summary_rows = build_summary_rows(metrics_by_filter)
    summary_csv_path = output_dir / "filter_stability_raw_metrics_summary.csv"
    save_summary_csv(summary_rows, summary_csv_path)
    raw_metrics_png_path, raw_metrics_svg_path = create_figure(
        metrics_by_filter,
        summary_rows,
        output_dir,
    )
    improvement_png_path, improvement_svg_path = create_improvement_figure(
        summary_rows,
        output_dir,
    )

    ranking = " > ".join(row["filter"] for row in summary_rows)
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved raw-metrics PNG figure: {raw_metrics_png_path}")
    print(f"Saved raw-metrics SVG figure: {raw_metrics_svg_path}")
    print(f"Saved improvement PNG figure: {improvement_png_path}")
    print(f"Saved improvement SVG figure: {improvement_svg_path}")
    print(f"Ranking: {ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
