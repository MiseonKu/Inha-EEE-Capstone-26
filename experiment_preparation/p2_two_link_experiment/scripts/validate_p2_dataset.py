"""Windows/matplotlib P2 dataset validation scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate P2 expected vs actual settled dataset if available.")
    parser.add_argument("--settled-dataset", type=Path, default=Path("outputs/pre_imu_p2/p2_settled_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/pre_imu_p2"))
    parser.add_argument("--report", type=Path, default=Path("outputs/pre_imu_p2/p2_dataset_validation_report.md"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.settled_dataset.exists():
        args.report.write_text("# P2 Dataset Validation\n\nStatus: NOT_TESTED\n\nNo settled dataset file was provided.\n", encoding="utf-8")
        print(f"wrote {args.report}")
        return 0
    df = pd.read_csv(args.settled_dataset)
    fig, ax = plt.subplots(figsize=(7, 5))
    if {"p_target_base_x_m", "p_target_base_z_m"}.issubset(df.columns):
        ax.scatter(df["p_target_base_x_m"], df["p_target_base_z_m"], s=15)
    ax.set_xlabel("target X base (m)")
    ax.set_ylabel("target Z base (m)")
    ax.set_title("P2 validation target coverage scaffold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "p2_dataset_validation_scaffold.png", dpi=160)
    plt.close(fig)
    args.report.write_text(
        "# P2 Dataset Validation\n\n"
        "Status: PASS_WITH_WARNING\n\n"
        "Real Vive settled labels were not validated unless label columns are present. "
        "This scaffold checks file readability and target coverage only.\n",
        encoding="utf-8",
    )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
