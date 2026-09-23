from __future__ import annotations

"""GPU-only clean_dataset_1link_v2 Kernel Ridge robustness evaluator.

This is the v2 pooled/session-robust counterpart of
``evaluate_v1000_gpu_kernel_ridge.py``. It refuses CPU fallback and evaluates
the same CUDA RBF Kernel Ridge baseline under several split policies:

- pooled_random: random 70/15/15 split over all v2 rows
- source_stratified: random 70/15/15 split preserving source_group ratio
- v1000_only_random: random 70/15/15 split over only v1000 rows
- v300_only_random: random 70/15/15 split over only v300 rows
- v1000_downsampled_to_v300: random 297-row v1000 subset, then 70/15/15
- holdout_v1000_to_v300: train/valid on v1000, test on v300
- holdout_v300_to_v1000: train/valid on v300, test on v1000
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from evaluate_v1000_gpu_kernel_ridge import (
    CudaKernelRidge,
    engineer_features,
    error_stats,
    parse_floats,
    parse_seeds,
    require_cuda,
    selection_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_gpu_krr_robustness"

V300_SOURCE = "test-260523-2"
V1000_SOURCE = "test-260524-1"


def git_commit_or_archive() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "archive_no_git_metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--alphas", default="0.0001,0.0003,0.001,0.003,0.01")
    parser.add_argument("--gammas", default="0.03,0.05,0.07,0.1,0.15,0.2")
    parser.add_argument(
        "--selection-metric",
        choices=("mean", "p95", "mean_plus_0p1_p95"),
        default="mean_plus_0p1_p95",
    )
    parser.add_argument(
        "--split-modes",
        default="pooled_random,source_stratified,holdout_v1000_to_v300,holdout_v300_to_v1000",
        help="Comma-separated split modes to run.",
    )
    parser.add_argument(
        "--include-source-one-hot",
        action="store_true",
        help=(
            "Diagnostic option: add source_group one-hot columns as features. "
            "Use this to test whether remaining pooled error is source-specific; "
            "do not treat it as source-holdout generalization."
        ),
    )
    parser.add_argument("--obsidian-assets", type=Path, default=None)
    parser.add_argument(
        "--asset-prefix",
        default=None,
        help="Optional prefix for copied Obsidian figure assets. Defaults to run-name.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nvidia_smi_text() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        return f"nvidia-smi failed: {exc!r}"


def load_data(dataset_root: Path, *, include_source_one_hot: bool) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    df = pd.read_csv(dataset_root / "dataset.csv")
    features = read_lines(dataset_root / "feature_columns.txt")
    labels = read_lines(dataset_root / "label_columns.txt")
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    quality = (
        (df["quality_flags"] == "ok")
        & df["settling_pass"].astype(bool)
        & df["target_id"].str.startswith("T")
        & df["label_tracker_valid"].astype(bool)
        & (~df["label_tracker_occluded"].astype(bool))
    )
    clean = df.loc[quality].copy().reset_index(drop=True)
    if include_source_one_hot:
        clean["feature_source_is_v300"] = (clean["source_group"] == V300_SOURCE).astype(float)
        clean["feature_source_is_v1000"] = (clean["source_group"] == V1000_SOURCE).astype(float)
        features = [*features, "feature_source_is_v300", "feature_source_is_v1000"]
    data_manifest = {
        "dataset_root": str(dataset_root),
        "raw_rows": int(len(df)),
        "filtered_rows": int(len(clean)),
        "source_counts": clean["source_group"].value_counts().to_dict(),
        "feature_columns": features,
        "label_columns": labels,
        "schema_version": manifest.get("schema_version"),
        "coordinate_policy": manifest.get("coordinate_policy", {}),
        "include_source_one_hot": include_source_one_hot,
    }
    return clean, features, labels, data_manifest


def _bundle(df: pd.DataFrame, features: list[str], labels: list[str], split: str) -> dict[str, Any]:
    meta_cols = [
        c
        for c in [
            "source_group",
            "source_session",
            "session_id",
            "target_id",
            "planned_order_index",
            "timestamp_utc",
        ]
        if c in df.columns
    ]
    return {
        "split": split,
        "X": engineer_features(df[features]),
        "y": df[labels].astype(float),
        "meta": df[meta_cols].copy(),
    }


def split_data(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seed: int,
    split_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if split_mode in {"pooled_random", "source_stratified", "v1000_only_random", "v300_only_random", "v1000_downsampled_to_v300"}:
        split_df = df
        if split_mode == "v1000_only_random":
            split_df = df.loc[df["source_group"] == V1000_SOURCE].copy()
        elif split_mode == "v300_only_random":
            split_df = df.loc[df["source_group"] == V300_SOURCE].copy()
        elif split_mode == "v1000_downsampled_to_v300":
            v300_rows = int((df["source_group"] == V300_SOURCE).sum())
            split_df = (
                df.loc[df["source_group"] == V1000_SOURCE]
                .sample(n=v300_rows, random_state=seed, replace=False)
                .copy()
            )

        stratify = split_df["source_group"] if split_mode == "source_stratified" else None
        train_df, temp_df = train_test_split(
            split_df,
            train_size=0.70,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        stratify_temp = temp_df["source_group"] if split_mode == "source_stratified" else None
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=seed,
            shuffle=True,
            stratify=stratify_temp,
        )
        return _bundle(train_df, features, labels, "train"), _bundle(valid_df, features, labels, "valid"), _bundle(test_df, features, labels, "test")

    if split_mode == "holdout_v1000_to_v300":
        train_source, test_source = V1000_SOURCE, V300_SOURCE
    elif split_mode == "holdout_v300_to_v1000":
        train_source, test_source = V300_SOURCE, V1000_SOURCE
    else:
        raise ValueError(f"unsupported split_mode: {split_mode}")

    train_pool = df.loc[df["source_group"] == train_source].copy()
    test_df = df.loc[df["source_group"] == test_source].copy()
    train_df, valid_df = train_test_split(train_pool, train_size=0.85, random_state=seed, shuffle=True)
    return _bundle(train_df, features, labels, "train"), _bundle(valid_df, features, labels, "valid"), _bundle(test_df, features, labels, "test")


def evaluate(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seeds: list[int],
    alphas: list[float],
    gammas: list[float],
    split_modes: list[str],
    *,
    selection_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    for split_mode in split_modes:
        for seed in seeds:
            train, valid, test = split_data(df, features, labels, seed, split_mode)
            best: tuple[float, float, float, CudaKernelRidge] | None = None
            for alpha in alphas:
                for gamma in gammas:
                    model = CudaKernelRidge(alpha=alpha, gamma=gamma).fit(train["X"], train["y"])
                    pred_valid = model.predict(valid["X"])
                    valid_stats, _ = error_stats(valid["y"], pred_valid)
                    score = selection_score(valid_stats, selection_metric)
                    grid_rows.append(
                        {
                            "split_mode": split_mode,
                            "seed": seed,
                            "alpha": alpha,
                            "gamma": gamma,
                            "valid_mean_3d_error_mm": valid_stats["mean_3d_error_mm"],
                            "valid_p95_3d_error_mm": valid_stats["p95_3d_error_mm"],
                            "selection_metric": selection_metric,
                            "selection_score": score,
                            "device_used": model.device_used_,
                            "cuda_device_name": model.cuda_device_name_,
                        }
                    )
                    if best is None or score < best[0]:
                        best = (score, alpha, gamma, model)
            assert best is not None
            _, alpha, gamma, model = best
            for bundle in (train, valid, test):
                pred = model.predict(bundle["X"])
                stats, err = error_stats(bundle["y"], pred)
                source_counts = bundle["meta"]["source_group"].value_counts().to_dict() if "source_group" in bundle["meta"] else {}
                metric_rows.append(
                    {
                        "split_mode": split_mode,
                        "seed": seed,
                        "model_name": "cuda_kernel_ridge_rbf_engineered",
                        "split": bundle["split"],
                        "rows": len(bundle["X"]),
                        "source_counts": json.dumps(source_counts, ensure_ascii=False),
                        "selected_alpha": alpha,
                        "selected_gamma": gamma,
                        "device_used": model.device_used_,
                        "cuda_device_name": model.cuda_device_name_,
                        **stats,
                    }
                )
                if bundle["split"] == "test":
                    p = bundle["meta"].reset_index(drop=True).copy()
                    p["split_mode"] = split_mode
                    p["seed"] = seed
                    p["model_name"] = "cuda_kernel_ridge_rbf_engineered"
                    p["split"] = "test"
                    p["error_3d_mm"] = err
                    for i, label in enumerate(labels):
                        p[f"true_{label}"] = bundle["y"].reset_index(drop=True)[label].to_numpy()
                        p[f"pred_{label}"] = pred[:, i]
                        p[f"error_{label}"] = pred[:, i] - p[f"true_{label}"]
                    pred_rows.append(p)
            torch.cuda.synchronize()
    return pd.DataFrame(metric_rows), pd.DataFrame(grid_rows), pd.concat(pred_rows, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mean_3d_error_mm",
        "median_3d_error_mm",
        "p90_3d_error_mm",
        "p95_3d_error_mm",
        "max_3d_error_mm",
        "pass_2mm_rate",
        "pass_2p5mm_rate",
    ]
    out = metrics.groupby(["split_mode", "model_name", "split"], as_index=False)[cols].agg(["mean", "std", "min", "max"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.reset_index(drop=True)


def plot_and_copy(
    metrics: pd.DataFrame,
    preds: pd.DataFrame,
    output_root: Path,
    obsidian_assets: Path | None,
    *,
    asset_prefix: str,
) -> dict[str, str]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    plt.style.use("seaborn-v0_8-whitegrid")

    test = metrics.loc[metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for mode, group in test.groupby("split_mode"):
        ax.plot(group["seed"], group["mean_3d_error_mm"], marker="o", label=mode)
    ax.axhline(2.5, color="tab:red", linestyle="--", label="2.5 mm")
    ax.axhline(2.0, color="tab:gray", linestyle=":", label="2.0 mm")
    ax.set_title("clean_dataset_1link_v2 CUDA KRR robustness by split mode")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test mean 3D error (mm)")
    ax.legend(fontsize=8)
    path = fig_dir / "v2_krr_split_mode_seed_test_mean.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["split_mode_seed_test_mean"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    order = list(test["split_mode"].drop_duplicates())
    ax.boxplot([test.loc[test["split_mode"] == mode, "mean_3d_error_mm"] for mode in order], tick_labels=order, showmeans=True)
    ax.axhline(2.5, color="tab:red", linestyle="--", label="2.5 mm")
    ax.set_title("Test mean 3D error distribution by split mode")
    ax.set_ylabel("Mean 3D error (mm)")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    path = fig_dir / "v2_krr_split_mode_boxplot.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["split_mode_boxplot"] = str(path)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for mode, group in preds.groupby("split_mode"):
        group["error_3d_mm"].plot(kind="hist", bins=35, alpha=0.35, ax=ax, label=mode)
    ax.axvline(2.5, color="tab:red", linestyle="--", label="2.5 mm")
    ax.set_title("Test error histogram by split mode")
    ax.set_xlabel("3D error (mm)")
    ax.legend(fontsize=8)
    path = fig_dir / "v2_krr_split_mode_error_hist.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["split_mode_error_hist"] = str(path)

    if obsidian_assets is not None:
        obsidian_assets.mkdir(parents=True, exist_ok=True)
        for key, src in list(written.items()):
            dst = obsidian_assets / f"{asset_prefix}_{Path(src).name}"
            shutil.copy2(src, dst)
            written[f"obsidian_{key}"] = str(dst)
    return written


def main() -> None:
    args = parse_args()
    gpu_status = require_cuda()
    seeds = parse_seeds(args.seeds)
    alphas = parse_floats(args.alphas)
    gammas = parse_floats(args.gammas)
    split_modes = [x.strip() for x in args.split_modes.split(",") if x.strip()]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    asset_prefix = args.asset_prefix or run_name
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "nvidia_smi_before.txt").write_text(nvidia_smi_text(), encoding="utf-8")

    df, features, labels, data_manifest = load_data(args.dataset_root, include_source_one_hot=args.include_source_one_hot)
    metrics, grid, preds = evaluate(
        df,
        features,
        labels,
        seeds,
        alphas,
        gammas,
        split_modes,
        selection_metric=args.selection_metric,
    )
    summary = summarize(metrics)
    figures = plot_and_copy(metrics, preds, out, args.obsidian_assets, asset_prefix=asset_prefix)

    metrics.to_csv(out / "seed_split_metrics.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out / "validation_grid_scores.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "repeated_summary.csv", index=False, encoding="utf-8-sig")

    best_rows = summary[summary["split"] == "test"].sort_values(["mean_3d_error_mm_mean"])
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_or_archive(),
        "gpu_status": gpu_status,
        "gpu_only_assertion": "CUDA required; CPU fallback is not implemented in this evaluator.",
        "data": data_manifest,
        "split_policy": "pooled/source-stratified 70/15/15, source holdout with train-source 85/15 train/valid",
        "selection_policy": f"alpha/gamma selected by validation {args.selection_metric}; test is final holdout.",
        "selection_metric": args.selection_metric,
        "split_modes": split_modes,
        "include_source_one_hot": args.include_source_one_hot,
        "asset_prefix": asset_prefix,
        "seeds": seeds,
        "alphas": alphas,
        "gammas": gammas,
        "outputs": {
            "metrics": str(out / "seed_split_metrics.csv"),
            "summary": str(out / "repeated_summary.csv"),
            "validation_grid": str(out / "validation_grid_scores.csv"),
            "predictions": str(out / "test_predictions.csv"),
            "figures": figures,
        },
        "best_test_summary": best_rows.iloc[0].to_dict() if len(best_rows) else {},
        "test_summary_by_split_mode": best_rows.to_dict(orient="records"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "gpu_status": gpu_status, "best_test_summary": manifest["best_test_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
