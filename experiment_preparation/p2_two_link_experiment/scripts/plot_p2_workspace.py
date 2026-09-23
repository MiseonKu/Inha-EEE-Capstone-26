"""Create matplotlib-only P2 workspace and trajectory validation plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot P2 target workspace and trajectories.")
    parser.add_argument("--targets", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_targets_pilot.csv")
    parser.add_argument("--trajectories", type=Path, default=ROOT / "outputs" / "pre_imu_p2" / "p2_trajectories.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots" / "pre_imu_p2")
    args = parser.parse_args()
    targets = pd.read_csv(args.targets)
    trajectories = pd.read_csv(args.trajectories)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(targets["p_target_base_x_m"], targets["p_target_base_y_m"], targets["p_target_base_z_m"], c=targets["azimuth_deg"], s=28)
    ax.set_xlabel("X base (m)")
    ax.set_ylabel("Y base (m)")
    ax.set_zlabel("Z base (m)")
    ax.set_title("P2 target workspace 3D")
    save(fig, out / "p2_workspace_3d.png")

    for x_col, y_col, name, x_label, y_label in [
        ("p_target_base_x_m", "p_target_base_y_m", "p2_workspace_xy.png", "X base (m)", "Y base (m)"),
        ("p_target_base_x_m", "p_target_base_z_m", "p2_workspace_xz.png", "X base (m)", "Z base (m)"),
        ("p_target_base_y_m", "p_target_base_z_m", "p2_workspace_yz.png", "Y base (m)", "Z base (m)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(targets[x_col], targets[y_col], s=25)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.set_title(name.replace(".png", ""))
        save(fig, out / name)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axs, ["q1_deg", "q2_deg", "q3_deg"], ["q1 yaw (deg)", "q2 shoulder (deg)", "q3 elbow (deg)"]):
        ax.hist(targets[col], bins=12)
        ax.set_title(title)
        ax.set_xlabel("deg")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
    save(fig, out / "p2_q_space.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = targets.groupby(["azimuth_bin", "radial_bin"]).size().sort_index()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("target count")
    ax.set_title("P2 selected target bin counts")
    save(fig, out / "p2_target_bins.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    preview = trajectories[trajectories["trial_id"] == trajectories["trial_id"].iloc[0]]
    ax.plot(preview["t_s"], preview["q1_deg"], label="q1 deg")
    ax.plot(preview["t_s"], preview["q2_deg"], label="q2 deg")
    ax.plot(preview["t_s"], preview["q3_deg"], label="q3 deg")
    ax.set_xlabel("time in trial (s)")
    ax.set_ylabel("joint angle (deg)")
    ax.set_title("P2 first-trial trajectory preview")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out / "p2_trajectory_preview.png")

    print(f"wrote plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
