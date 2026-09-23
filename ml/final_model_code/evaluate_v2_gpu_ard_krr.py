from __future__ import annotations

"""GPU-only ARD/group-scaled RBF Kernel Ridge evaluator for v2.

The previous runs showed that plain RBF KRR is the strongest model family, but
it uses a single isotropic gamma after standardization. This script tests
hand-designed feature-group scale profiles (an ARD-like diagonal metric) while
keeping the fit/solve on CUDA tensors.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_v1000_gpu_kernel_ridge import engineer_features, error_stats, parse_floats, parse_seeds, require_cuda, selection_score
from evaluate_v2_gpu_krr_robustness import DATASET_ROOT, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_gpu_ard_krr"


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
    parser.add_argument("--gammas", default="0.01,0.03,0.05,0.07,0.1,0.15,0.2")
    parser.add_argument("--split-modes", default="v1000_only_random,v300_only_random,source_stratified")
    parser.add_argument("--selection-metric", choices=("mean", "p95", "mean_plus_0p1_p95"), default="mean_plus_0p1_p95")
    parser.add_argument(
        "--profile-names",
        default="",
        help="Optional comma-separated subset of profile names from the catalog.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data(dataset_root: Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    df = pd.read_csv(dataset_root / "dataset.csv")
    features = read_lines(dataset_root / "feature_columns.txt")
    labels = read_lines(dataset_root / "label_columns.txt")
    quality = (
        (df["quality_flags"] == "ok")
        & df["settling_pass"].astype(bool)
        & df["target_id"].str.startswith("T")
        & df["label_tracker_valid"].astype(bool)
        & (~df["label_tracker_occluded"].astype(bool))
    )
    clean = df.loc[quality].copy().reset_index(drop=True)
    return clean, features, labels, {
        "dataset_root": str(dataset_root),
        "rows": int(len(clean)),
        "source_counts": clean["source_group"].value_counts().to_dict(),
        "feature_columns": features,
        "label_columns": labels,
    }


def profile_catalog() -> list[dict[str, float]]:
    # These profiles scale standardized feature groups before RBF distance.
    # Values >1 make the kernel more sensitive to that group.
    profiles: list[dict[str, float]] = [{"name": "iso"}]
    for target in [0.5, 1.5, 2.0, 3.0]:
        profiles.append({"name": f"target_x{target:g}", "target": target})
    for angles in [0.5, 1.5, 2.0, 3.0]:
        profiles.append({"name": f"angles_x{angles:g}", "angles": angles})
    for trig in [0.5, 1.5, 2.0, 3.0]:
        profiles.append({"name": f"trig_x{trig:g}", "trig": trig})
    for delta in [0.5, 1.5, 2.0, 3.0]:
        profiles.append({"name": f"delta_x{delta:g}", "delta": delta})
    combos = [
        ("target2_delta2", {"target": 2.0, "delta": 2.0}),
        ("target2_angles05", {"target": 2.0, "angles": 0.5}),
        ("target05_trig2", {"target": 0.5, "trig": 2.0}),
        ("angles2_trig05", {"angles": 2.0, "trig": 0.5}),
        ("target3_delta05", {"target": 3.0, "delta": 0.5}),
        ("target05_delta3", {"target": 0.5, "delta": 3.0}),
        ("angles05_target2_delta2", {"angles": 0.5, "target": 2.0, "delta": 2.0}),
        ("angles2_target05_trig2", {"angles": 2.0, "target": 0.5, "trig": 2.0}),
    ]
    for name, values in combos:
        profiles.append({"name": name, **values})
    return profiles


def profile_vector(columns: list[str], profile: dict[str, float]) -> np.ndarray:
    scale = np.ones(len(columns), dtype=np.float64)
    for i, col in enumerate(columns):
        if col in {"q1_cmd_deg", "q2_cmd_deg", "q1_encoder_deg", "q2_encoder_deg"}:
            scale[i] *= profile.get("angles", 1.0)
        if col.startswith("p_target_") or col == "target_r_mm":
            scale[i] *= profile.get("target", 1.0)
        if col.endswith("_sin") or col.endswith("_cos"):
            scale[i] *= profile.get("trig", 1.0)
        if col.endswith("_minus_encoder"):
            scale[i] *= profile.get("delta", 1.0)
    return scale


class CudaArdKRR:
    def __init__(self, *, alpha: float, gamma: float, profile: dict[str, float]) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.profile = profile
        self.device = torch.device("cuda")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CudaArdKRR":
        X_np = X.to_numpy(dtype=np.float64)
        y_np = y.to_numpy(dtype=np.float64)
        self.columns_ = list(X.columns)
        self.profile_scale_ = profile_vector(self.columns_, self.profile)
        self.x_median_ = np.nanmedian(X_np, axis=0)
        X_np = np.where(np.isfinite(X_np), X_np, self.x_median_)
        self.x_mean_ = X_np.mean(axis=0)
        self.x_std_ = X_np.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        X_scaled = ((X_np - self.x_mean_) / self.x_std_) * self.profile_scale_
        self.X_train_ = torch.as_tensor(X_scaled, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y_np, dtype=torch.float64, device=self.device)
        K = torch.exp(-self.gamma * (torch.cdist(self.X_train_, self.X_train_) ** 2))
        eye = torch.eye(K.shape[0], dtype=torch.float64, device=self.device)
        self.dual_coef_ = torch.linalg.solve(K + eye * self.alpha, y_t)
        self.device_used_ = "cuda"
        self.cuda_device_name_ = torch.cuda.get_device_name(0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.to_numpy(dtype=np.float64)
        X_np = np.where(np.isfinite(X_np), X_np, self.x_median_)
        X_scaled = ((X_np - self.x_mean_) / self.x_std_) * self.profile_scale_
        X_t = torch.as_tensor(X_scaled, dtype=torch.float64, device=self.device)
        K = torch.exp(-self.gamma * (torch.cdist(X_t, self.X_train_) ** 2))
        return (K @ self.dual_coef_).detach().cpu().numpy()


def evaluate(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seeds: list[int],
    alphas: list[float],
    gammas: list[float],
    split_modes: list[str],
    selection_metric: str,
    profile_names: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    profiles = profile_catalog()
    if profile_names:
        profiles = [profile for profile in profiles if profile["name"] in profile_names]
        missing = sorted(profile_names.difference({profile["name"] for profile in profiles}))
        if missing:
            raise ValueError(f"unknown profile names: {missing}")
    for split_mode in split_modes:
        for seed in seeds:
            train, valid, test = split_data(df, features, labels, seed, split_mode)
            best: tuple[float, dict[str, Any], CudaArdKRR] | None = None
            for profile in profiles:
                for alpha in alphas:
                    for gamma in gammas:
                        model = CudaArdKRR(alpha=alpha, gamma=gamma, profile=profile).fit(train["X"], train["y"])
                        pred_valid = model.predict(valid["X"])
                        valid_stats, _ = error_stats(valid["y"], pred_valid)
                        score = selection_score(valid_stats, selection_metric)
                        grid_rows.append(
                            {
                                "split_mode": split_mode,
                                "seed": seed,
                                "profile_name": profile["name"],
                                "alpha": alpha,
                                "gamma": gamma,
                                "valid_mean_3d_error_mm": valid_stats["mean_3d_error_mm"],
                                "valid_p95_3d_error_mm": valid_stats["p95_3d_error_mm"],
                                "selection_score": score,
                            }
                        )
                        if best is None or score < best[0]:
                            best = (score, {"profile_name": profile["name"], "alpha": alpha, "gamma": gamma, "profile": profile}, model)
            assert best is not None
            _, params, model = best
            for bundle in (train, valid, test):
                pred = model.predict(bundle["X"])
                stats, _ = error_stats(bundle["y"], pred)
                metric_rows.append(
                    {
                        "split_mode": split_mode,
                        "seed": seed,
                        "model_name": "cuda_ard_rbf_krr",
                        "split": bundle["split"],
                        "rows": len(bundle["X"]),
                        "selected_profile": params["profile_name"],
                        "selected_alpha": params["alpha"],
                        "selected_gamma": params["gamma"],
                        "device_used": model.device_used_,
                        "cuda_device_name": model.cuda_device_name_,
                        **stats,
                    }
                )
            torch.cuda.synchronize()
    return pd.DataFrame(metric_rows), pd.DataFrame(grid_rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["mean_3d_error_mm", "median_3d_error_mm", "p90_3d_error_mm", "p95_3d_error_mm", "max_3d_error_mm", "pass_2mm_rate", "pass_2p5mm_rate"]
    out = metrics.groupby(["split_mode", "model_name", "split"], as_index=False)[cols].agg(["mean", "std", "min", "max"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    gpu_status = require_cuda()
    seeds = parse_seeds(args.seeds)
    alphas = parse_floats(args.alphas)
    gammas = parse_floats(args.gammas)
    split_modes = [x.strip() for x in args.split_modes.split(",") if x.strip()]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    df, features, labels, data_manifest = load_data(args.dataset_root)
    profile_names = {x.strip() for x in args.profile_names.split(",") if x.strip()} or None
    metrics, grid = evaluate(df, features, labels, seeds, alphas, gammas, split_modes, args.selection_metric, profile_names=profile_names)
    summary = summarize(metrics)
    metrics.to_csv(out / "seed_split_metrics.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out / "validation_grid_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "repeated_summary.csv", index=False, encoding="utf-8-sig")
    best_rows = summary[summary["split"] == "test"].sort_values("mean_3d_error_mm_mean")
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_or_archive(),
        "gpu_status": gpu_status,
        "gpu_only_assertion": "CUDA tensors are required for all ARD KRR kernel matrices and solves.",
        "data": data_manifest,
        "split_modes": split_modes,
        "profiles": profile_catalog(),
        "profile_names_filter": sorted(profile_names) if profile_names else None,
        "alphas": alphas,
        "gammas": gammas,
        "selection_metric": args.selection_metric,
        "outputs": {
            "metrics": str(out / "seed_split_metrics.csv"),
            "validation_grid": str(out / "validation_grid_scores.csv"),
            "summary": str(out / "repeated_summary.csv"),
        },
        "best_test_summary": best_rows.iloc[0].to_dict() if len(best_rows) else {},
        "test_summary_by_split_mode": best_rows.to_dict(orient="records"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "gpu_status": gpu_status, "best_test_summary": manifest["best_test_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
