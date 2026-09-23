from __future__ import annotations

"""GPU-only v1000 single-session Kernel Ridge evaluator.

This script is the GPU-valid counterpart of the CPU diagnostic baseline.
It refuses to run unless PyTorch CUDA is available and all kernel fitting /
prediction tensors live on `cuda`.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_1link_v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v1000_gpu_kernel_ridge"
SOURCE_GROUP = "test-260524-1"
SOURCE_SESSION = "rigid2_20260523_140957Z_v1000_1000t"


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
        default="mean",
        help="Validation metric used to select alpha/gamma for each seed.",
    )
    parser.add_argument("--obsidian-assets", type=Path, default=None)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. torch.cuda.is_available() is false; aborting CPU fallback.")
    torch.cuda.set_device(0)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": True,
        "cuda_device_index": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": tuple(int(v) for v in torch.cuda.get_device_capability(0)),
    }


def nvidia_smi_text() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover
        return f"nvidia-smi failed: {exc!r}"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data(dataset_root: Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    df = pd.read_csv(dataset_root / "dataset.csv")
    features = read_lines(dataset_root / "feature_columns.txt")
    labels = read_lines(dataset_root / "label_columns.txt")
    mask = (df["source_group"] == SOURCE_GROUP) & (df["source_session"] == SOURCE_SESSION)
    session = df.loc[mask].copy().reset_index(drop=True)
    quality = (
        (session["quality_flags"] == "ok")
        & session["settling_pass"].astype(bool)
        & session["target_id"].str.startswith("T")
        & session["label_tracker_valid"].astype(bool)
        & (~session["label_tracker_occluded"].astype(bool))
    )
    clean = session.loc[quality].copy().reset_index(drop=True)
    manifest = {
        "dataset_root": str(dataset_root),
        "source_group": SOURCE_GROUP,
        "source_session": SOURCE_SESSION,
        "raw_session_rows": int(len(session)),
        "filtered_rows": int(len(clean)),
        "feature_columns": features,
        "label_columns": labels,
    }
    return clean, features, labels, manifest


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.astype(float).copy()
    for col in ["q1_cmd_deg", "q2_cmd_deg", "q1_encoder_deg", "q2_encoder_deg"]:
        radians = np.deg2rad(out[col].to_numpy(dtype=float))
        out[f"{col}_sin"] = np.sin(radians)
        out[f"{col}_cos"] = np.cos(radians)
    out["q1_cmd_minus_encoder"] = out["q1_cmd_deg"] - out["q1_encoder_deg"]
    out["q2_cmd_minus_encoder"] = out["q2_cmd_deg"] - out["q2_encoder_deg"]
    out["target_r_mm"] = np.linalg.norm(out[["p_target_x_mm", "p_target_y_mm", "p_target_z_mm"]].to_numpy(dtype=float), axis=1)
    return out


def split_data(df: pd.DataFrame, features: list[str], labels: list[str], seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    X = engineer_features(df[features])
    y = df[labels].astype(float)
    meta_cols = [c for c in ["source_group", "source_session", "session_id", "target_id", "planned_order_index"] if c in df.columns]
    meta = df[meta_cols].copy()
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, meta, train_size=0.70, random_state=seed, shuffle=True
    )
    X_valid, X_test, y_valid, y_test, meta_valid, meta_test = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.50, random_state=seed, shuffle=True
    )
    return (
        {"split": "train", "X": X_train, "y": y_train, "meta": meta_train},
        {"split": "valid", "X": X_valid, "y": y_valid, "meta": meta_valid},
        {"split": "test", "X": X_test, "y": y_test, "meta": meta_test},
    )


class CudaKernelRidge:
    def __init__(self, *, alpha: float, gamma: float) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.device = torch.device("cuda")

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "CudaKernelRidge":
        X_np = X.to_numpy(dtype=np.float64)
        y_np = y.to_numpy(dtype=np.float64)
        self.x_median_ = np.nanmedian(X_np, axis=0)
        X_np = np.where(np.isfinite(X_np), X_np, self.x_median_)
        self.x_mean_ = X_np.mean(axis=0)
        self.x_std_ = X_np.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        X_scaled = (X_np - self.x_mean_) / self.x_std_
        self.X_train_ = torch.as_tensor(X_scaled, dtype=torch.float64, device=self.device)
        y_t = torch.as_tensor(y_np, dtype=torch.float64, device=self.device)
        K = self._rbf(self.X_train_, self.X_train_)
        eye = torch.eye(K.shape[0], dtype=torch.float64, device=self.device)
        self.dual_coef_ = torch.linalg.solve(K + eye * self.alpha, y_t)
        self.device_used_ = str(self.device)
        self.cuda_device_name_ = torch.cuda.get_device_name(0)
        return self

    def _rbf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (torch.cdist(a, b) ** 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = X.to_numpy(dtype=np.float64)
        X_np = np.where(np.isfinite(X_np), X_np, self.x_median_)
        X_scaled = (X_np - self.x_mean_) / self.x_std_
        X_t = torch.as_tensor(X_scaled, dtype=torch.float64, device=self.device)
        pred = self._rbf(X_t, self.X_train_) @ self.dual_coef_
        return pred.detach().cpu().numpy()


def error_stats(y_true: pd.DataFrame, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    err = np.linalg.norm(y_pred - y_true.to_numpy(), axis=1)
    return (
        {
            "mean_3d_error_mm": float(np.mean(err)),
            "median_3d_error_mm": float(np.median(err)),
            "p90_3d_error_mm": float(np.percentile(err, 90)),
            "p95_3d_error_mm": float(np.percentile(err, 95)),
            "max_3d_error_mm": float(np.max(err)),
            "pass_2mm_rate": float(np.mean(err <= 2.0)),
            "pass_2p5mm_rate": float(np.mean(err <= 2.5)),
        },
        err,
    )


def selection_score(stats: dict[str, float], metric: str) -> float:
    if metric == "mean":
        return stats["mean_3d_error_mm"]
    if metric == "p95":
        return stats["p95_3d_error_mm"]
    if metric == "mean_plus_0p1_p95":
        return stats["mean_3d_error_mm"] + 0.1 * stats["p95_3d_error_mm"]
    raise ValueError(f"unsupported selection metric: {metric}")


def evaluate(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seeds: list[int],
    alphas: list[float],
    gammas: list[float],
    *,
    selection_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    for seed in seeds:
        train, valid, test = split_data(df, features, labels, seed)
        best: tuple[float, float, float, CudaKernelRidge] | None = None
        for alpha in alphas:
            for gamma in gammas:
                model = CudaKernelRidge(alpha=alpha, gamma=gamma).fit(train["X"], train["y"])
                pred_valid = model.predict(valid["X"])
                valid_stats, _ = error_stats(valid["y"], pred_valid)
                score = selection_score(valid_stats, selection_metric)
                grid_rows.append(
                    {
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
            metric_rows.append({"seed": seed, "model_name": "cuda_kernel_ridge_rbf_engineered", "split": bundle["split"], "rows": len(bundle["X"]), "selected_alpha": alpha, "selected_gamma": gamma, "device_used": model.device_used_, "cuda_device_name": model.cuda_device_name_, **stats})
            if bundle["split"] == "test":
                p = bundle["meta"].reset_index(drop=True).copy()
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
    cols = ["mean_3d_error_mm", "median_3d_error_mm", "p90_3d_error_mm", "p95_3d_error_mm", "max_3d_error_mm", "pass_2mm_rate", "pass_2p5mm_rate"]
    out = metrics.groupby(["model_name", "split"], as_index=False)[cols].agg(["mean", "std", "min", "max"])
    out.columns = ["_".join(c).rstrip("_") for c in out.columns.to_flat_index()]
    return out.reset_index(drop=True)


def plot_and_copy(metrics: pd.DataFrame, preds: pd.DataFrame, output_root: Path, obsidian_assets: Path | None) -> dict[str, str]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    test = metrics[metrics["split"] == "test"]
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.plot(test["seed"], test["mean_3d_error_mm"], marker="o", label="CUDA KernelRidge")
    ax.axhline(2.5, color="tab:red", linestyle="--", label="2.5 mm target")
    ax.axhline(2.0, color="tab:gray", linestyle=":", label="2.0 mm reference")
    ax.set_title("GPU-only v1000 repeated split test mean 3D error")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test mean 3D error (mm)")
    ax.legend()
    path = fig_dir / "gpu_v1000_seed_test_mean_error.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["seed_test_mean_error"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    preds["error_3d_mm"].plot(kind="hist", bins=30, ax=ax)
    ax.axvline(2.5, color="tab:red", linestyle="--", label="2.5 mm target")
    ax.set_title("GPU-only CUDA KernelRidge repeated test error distribution")
    ax.set_xlabel("3D error (mm)")
    ax.set_ylabel("Sample count")
    ax.legend()
    path = fig_dir / "gpu_v1000_test_error_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["test_error_distribution"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    for ax, label, axis in zip(axes, ["p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"], ["x", "y", "z"]):
        ax.scatter(preds[f"true_{label}"], preds[f"pred_{label}"], s=10, alpha=0.55)
        lo = min(preds[f"true_{label}"].min(), preds[f"pred_{label}"].min())
        hi = max(preds[f"true_{label}"].max(), preds[f"pred_{label}"].max())
        ax.plot([lo, hi], [lo, hi], color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(f"{axis}-axis")
        ax.set_xlabel("Ground truth (mm)")
        ax.set_ylabel("Prediction (mm)")
    fig.suptitle("GPU-only CUDA KernelRidge predicted vs ground truth")
    path = fig_dir / "gpu_v1000_pred_vs_true_axes.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["pred_vs_true_axes"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    order = ["train", "valid", "test"]
    ax.boxplot([metrics.loc[metrics["split"] == s, "mean_3d_error_mm"] for s in order], tick_labels=order, showmeans=True)
    ax.axhline(2.5, color="tab:red", linestyle="--", label="2.5 mm target")
    ax.set_title("GPU-only CUDA KernelRidge train/valid/test gap")
    ax.set_ylabel("Mean 3D error (mm)")
    ax.legend()
    path = fig_dir / "gpu_v1000_train_valid_test_gap.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written["train_valid_test_gap"] = str(path)

    if obsidian_assets is not None:
        obsidian_assets.mkdir(parents=True, exist_ok=True)
        for key, src in list(written.items()):
            dst = obsidian_assets / f"2026-05-29_{Path(src).name}"
            shutil.copy2(src, dst)
            written[f"obsidian_{key}"] = str(dst)
    return written


def main() -> None:
    args = parse_args()
    gpu_status = require_cuda()
    smi = nvidia_smi_text()
    seeds = parse_seeds(args.seeds)
    alphas = parse_floats(args.alphas)
    gammas = parse_floats(args.gammas)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "nvidia_smi_before.txt").write_text(smi, encoding="utf-8")

    df, features, labels, data_manifest = load_data(args.dataset_root)
    metrics, grid, preds = evaluate(
        df,
        features,
        labels,
        seeds,
        alphas,
        gammas,
        selection_metric=args.selection_metric,
    )
    summary = summarize(metrics)
    figures = plot_and_copy(metrics, preds, out, args.obsidian_assets)

    metrics.to_csv(out / "seed_split_metrics.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out / "validation_grid_scores.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "repeated_summary.csv", index=False, encoding="utf-8-sig")
    (out / "selected_params_by_seed.csv").write_text(
        metrics[metrics["split"] == "test"][["seed", "selected_alpha", "selected_gamma", "mean_3d_error_mm", "pass_2p5mm_rate", "device_used", "cuda_device_name"]].to_csv(index=False),
        encoding="utf-8-sig",
    )

    best = summary[summary["split"] == "test"].sort_values("mean_3d_error_mm_mean").iloc[0].to_dict()
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_or_archive(),
        "gpu_status": gpu_status,
        "gpu_only_assertion": "CUDA required; CPU fallback is not implemented in this evaluator.",
        "data": data_manifest,
        "split_policy": "repeated random split train/valid/test = 70/15/15",
        "selection_policy": f"alpha/gamma selected by validation {args.selection_metric}; test is final holdout.",
        "selection_metric": args.selection_metric,
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
        "best_test_summary": best,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "gpu_status": gpu_status, "best_test_summary": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
