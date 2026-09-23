from __future__ import annotations

"""GPU-only ARD KRR profile ensemble evaluator.

This tests whether averaging several validation-selected/fixed ARD RBF KRR
profiles can move the v1000-only result below 2 mm without using test labels
for selection.
"""

import argparse
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_v1000_gpu_kernel_ridge import error_stats, parse_seeds, require_cuda, selection_score
from evaluate_v2_gpu_ard_krr import CudaArdKRR, load_data, profile_catalog
from evaluate_v2_gpu_krr_robustness import DATASET_ROOT, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_gpu_ard_krr_ensemble"


FIXED_CANDIDATES: list[dict[str, Any]] = [
    {"name": "target3_delta05_a1e-4_g0p03", "profile_name": "target3_delta05", "alpha": 0.0001, "gamma": 0.03},
    {"name": "target3_delta05_a3e-4_g0p03", "profile_name": "target3_delta05", "alpha": 0.0003, "gamma": 0.03},
    {"name": "target_x3_a1e-4_g0p03", "profile_name": "target_x3", "alpha": 0.0001, "gamma": 0.03},
    {"name": "trig_x2_a3e-4_g0p05", "profile_name": "trig_x2", "alpha": 0.0003, "gamma": 0.05},
    {"name": "trig_x2_a1e-4_g0p03", "profile_name": "trig_x2", "alpha": 0.0001, "gamma": 0.03},
    {"name": "target05_trig2_a3e-4_g0p05", "profile_name": "target05_trig2", "alpha": 0.0003, "gamma": 0.05},
    {"name": "target_x2_a1e-4_g0p07", "profile_name": "target_x2", "alpha": 0.0001, "gamma": 0.07},
    {"name": "target_x2_a1e-3_g0p15", "profile_name": "target_x2", "alpha": 0.001, "gamma": 0.15},
]


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
    parser.add_argument("--split-mode", default="v1000_only_random")
    parser.add_argument("--selection-metric", choices=("mean", "p95", "mean_plus_0p1_p95"), default="mean")
    parser.add_argument("--max-ensemble-size", type=int, default=4)
    return parser.parse_args()


def candidate_profiles() -> dict[str, dict[str, float]]:
    return {profile["name"]: profile for profile in profile_catalog()}


def fit_candidate(candidate: dict[str, Any], train: dict[str, Any], profiles: dict[str, dict[str, float]]) -> CudaArdKRR:
    return CudaArdKRR(
        alpha=float(candidate["alpha"]),
        gamma=float(candidate["gamma"]),
        profile=profiles[str(candidate["profile_name"])],
    ).fit(train["X"], train["y"])


def combo_name(combo: tuple[int, ...]) -> str:
    return "+".join(FIXED_CANDIDATES[i]["name"] for i in combo)


def evaluate(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seeds: list[int],
    split_mode: str,
    selection_metric: str,
    max_ensemble_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = candidate_profiles()
    candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    combos: list[tuple[int, ...]] = []
    for size in range(1, max_ensemble_size + 1):
        combos.extend(itertools.combinations(range(len(FIXED_CANDIDATES)), size))

    for seed in seeds:
        train, valid, test = split_data(df, features, labels, seed, split_mode)
        valid_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        train_preds: list[np.ndarray] = []
        for i, candidate in enumerate(FIXED_CANDIDATES):
            model = fit_candidate(candidate, train, profiles)
            pred_train = model.predict(train["X"])
            pred_valid = model.predict(valid["X"])
            pred_test = model.predict(test["X"])
            train_preds.append(pred_train)
            valid_preds.append(pred_valid)
            test_preds.append(pred_test)
            valid_stats, _ = error_stats(valid["y"], pred_valid)
            candidate_rows.append({"seed": seed, "candidate_index": i, **candidate, **{f"valid_{k}": v for k, v in valid_stats.items()}})

        best: tuple[float, tuple[int, ...], dict[str, float]] | None = None
        for combo in combos:
            pred_valid = np.mean([valid_preds[i] for i in combo], axis=0)
            valid_stats, _ = error_stats(valid["y"], pred_valid)
            score = selection_score(valid_stats, selection_metric)
            grid_rows.append(
                {
                    "seed": seed,
                    "combo": combo_name(combo),
                    "ensemble_size": len(combo),
                    "selection_metric": selection_metric,
                    "selection_score": score,
                    **{f"valid_{k}": v for k, v in valid_stats.items()},
                }
            )
            if best is None or score < best[0]:
                best = (score, combo, valid_stats)
        assert best is not None
        _, selected_combo, _ = best
        for split, bundle, pred_bank in [("train", train, train_preds), ("valid", valid, valid_preds), ("test", test, test_preds)]:
            pred = np.mean([pred_bank[i] for i in selected_combo], axis=0)
            stats, err = error_stats(bundle["y"], pred)
            metric_rows.append(
                {
                    "split_mode": split_mode,
                    "seed": seed,
                    "model_name": "cuda_ard_krr_profile_ensemble",
                    "split": split,
                    "rows": len(bundle["X"]),
                    "selected_combo": combo_name(selected_combo),
                    "ensemble_size": len(selected_combo),
                    "device_used": "cuda",
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    **stats,
                }
            )
            if split == "test":
                p = bundle["meta"].reset_index(drop=True).copy()
                p["split_mode"] = split_mode
                p["seed"] = seed
                p["model_name"] = "cuda_ard_krr_profile_ensemble"
                p["selected_combo"] = combo_name(selected_combo)
                p["error_3d_mm"] = err
                for j, label in enumerate(labels):
                    p[f"true_{label}"] = bundle["y"].reset_index(drop=True)[label].to_numpy()
                    p[f"pred_{label}"] = pred[:, j]
                    p[f"error_{label}"] = pred[:, j] - p[f"true_{label}"]
                pred_rows.append(p)
        torch.cuda.synchronize()
    return pd.DataFrame(metric_rows), pd.DataFrame(grid_rows), pd.DataFrame(candidate_rows), pd.concat(pred_rows, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["mean_3d_error_mm", "median_3d_error_mm", "p90_3d_error_mm", "p95_3d_error_mm", "max_3d_error_mm", "pass_2mm_rate", "pass_2p5mm_rate"]
    out = metrics.groupby(["split_mode", "model_name", "split"], as_index=False)[cols].agg(["mean", "std", "min", "max"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    gpu_status = require_cuda()
    seeds = parse_seeds(args.seeds)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    df, features, labels, data_manifest = load_data(args.dataset_root)
    metrics, grid, candidate_scores, preds = evaluate(
        df,
        features,
        labels,
        seeds,
        args.split_mode,
        args.selection_metric,
        args.max_ensemble_size,
    )
    summary = summarize(metrics)
    metrics.to_csv(out / "seed_split_metrics.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out / "validation_ensemble_grid.csv", index=False, encoding="utf-8-sig")
    candidate_scores.to_csv(out / "validation_candidate_scores.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "repeated_summary.csv", index=False, encoding="utf-8-sig")
    best_rows = summary[summary["split"] == "test"].sort_values("mean_3d_error_mm_mean")
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_or_archive(),
        "gpu_status": gpu_status,
        "gpu_only_assertion": "All candidate ARD KRR models use CUDA tensors.",
        "data": data_manifest,
        "split_mode": args.split_mode,
        "selection_metric": args.selection_metric,
        "max_ensemble_size": args.max_ensemble_size,
        "fixed_candidates": FIXED_CANDIDATES,
        "outputs": {
            "metrics": str(out / "seed_split_metrics.csv"),
            "validation_ensemble_grid": str(out / "validation_ensemble_grid.csv"),
            "validation_candidate_scores": str(out / "validation_candidate_scores.csv"),
            "predictions": str(out / "test_predictions.csv"),
            "summary": str(out / "repeated_summary.csv"),
        },
        "best_test_summary": best_rows.iloc[0].to_dict() if len(best_rows) else {},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "gpu_status": gpu_status, "best_test_summary": manifest["best_test_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
