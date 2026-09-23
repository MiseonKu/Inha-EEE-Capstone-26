from __future__ import annotations

"""Evaluate the 1-Link final local-kernel method on clean_dataset_2link_v1."""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_2link_v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "by_date" / "2026-06-01-2link-v1-model-search" / "02_local_kernel_candidates"


@dataclass(frozen=True)
class SplitBundle:
    name: str
    x: pd.DataFrame
    y: pd.DataFrame
    meta: pd.DataFrame


def parse_seeds(raw: str) -> list[int]:
    if "-" in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this goal; aborting CPU-only run.")
    return torch.cuda.get_device_name(0)


def error_stats(y_true: pd.DataFrame, pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    true = y_true.to_numpy(dtype=np.float64)
    err_axis = pred - true
    err = np.linalg.norm(err_axis, axis=1)
    return {
        "mean_3d_error_mm": float(np.mean(err)),
        "median_3d_error_mm": float(np.median(err)),
        "p90_3d_error_mm": float(np.percentile(err, 90)),
        "p95_3d_error_mm": float(np.percentile(err, 95)),
        "max_3d_error_mm": float(np.max(err)),
        "pass_2mm_rate": float(np.mean(err <= 2.0)),
        "pass_2p5mm_rate": float(np.mean(err <= 2.5)),
    }, err


def split_frame(df: pd.DataFrame, features: list[str], labels: list[str], seed: int) -> tuple[SplitBundle, SplitBundle, SplitBundle]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    train_end = int(len(idx) * 0.70)
    valid_end = int(len(idx) * 0.85)
    bundles: list[SplitBundle] = []
    for name, part in [("train", idx[:train_end]), ("valid", idx[train_end:valid_end]), ("test", idx[valid_end:])]:
        split_df = df.iloc[part].reset_index(drop=True)
        meta_cols = [c for c in ["source_group", "source_session", "target_id", "sample_id", "planned_order_index"] if c in split_df]
        bundles.append(SplitBundle(name, split_df[features], split_df[labels], split_df[meta_cols]))
    return bundles[0], bundles[1], bundles[2]


def standardize(train: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_np = train.to_numpy(dtype=np.float64)
    query_np = query.to_numpy(dtype=np.float64)
    median = np.nanmedian(train_np, axis=0)
    train_np = np.where(np.isfinite(train_np), train_np, median)
    query_np = np.where(np.isfinite(query_np), query_np, median)
    mean = train_np.mean(axis=0)
    std = train_np.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train_np - mean) / std, (query_np - mean) / std


def fit_ridge(train: SplitBundle, query: SplitBundle, alpha: float) -> np.ndarray:
    x_train, x_query = standardize(train.x, query.x)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_query = np.c_[np.ones(len(x_query)), x_query]
    xtx = x_train.T @ x_train
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    xty = x_train.T @ train.y.to_numpy(dtype=np.float64)
    coef = np.linalg.solve(xtx + penalty, xty)
    return x_query @ coef


def fit_krr(train: SplitBundle, queries: list[SplitBundle], alpha: float, gamma: float) -> list[np.ndarray]:
    x_train, _ = standardize(train.x, train.x)
    train_t = torch.as_tensor(x_train, dtype=torch.float32, device="cuda")
    y_t = torch.as_tensor(train.y.to_numpy(dtype=np.float32), dtype=torch.float32, device="cuda")
    k_train = torch.exp(-gamma * torch.cdist(train_t, train_t).square())
    eye = torch.eye(k_train.shape[0], dtype=torch.float32, device="cuda")
    beta = torch.linalg.solve(k_train + alpha * eye, y_t)
    preds: list[np.ndarray] = []
    for query in queries:
        _, x_query = standardize(train.x, query.x)
        query_t = torch.as_tensor(x_query, dtype=torch.float32, device="cuda")
        k_query = torch.exp(-gamma * torch.cdist(query_t, train_t).square())
        preds.append((k_query @ beta).detach().cpu().numpy().astype(np.float64))
    return preds


def local_average(values: np.ndarray, train: SplitBundle, query: SplitBundle, k: int, scale: float) -> np.ndarray:
    x_train, x_query = standardize(train.x, query.x)
    train_t = torch.as_tensor(x_train, dtype=torch.float32, device="cuda")
    query_t = torch.as_tensor(x_query, dtype=torch.float32, device="cuda")
    dist = torch.cdist(query_t, train_t)
    k_eff = min(k, dist.shape[1])
    nn_dist, nn_idx = torch.topk(dist, k=k_eff, largest=False)
    bandwidth = torch.clamp(nn_dist[:, -1:] * scale, min=1e-6)
    weights = torch.exp(-0.5 * (nn_dist / bandwidth).square())
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
    values_t = torch.as_tensor(values, dtype=torch.float32, device="cuda")
    return torch.sum(values_t[nn_idx] * weights.unsqueeze(-1), dim=1).detach().cpu().numpy().astype(np.float64)


def evaluate_seed(df: pd.DataFrame, features: list[str], labels: list[str], seed: int, device_name: str) -> tuple[list[dict[str, float | int | str]], pd.DataFrame]:
    train, valid, test = split_frame(df, features, labels, seed)
    metrics: list[dict[str, float | int | str]] = []
    best: tuple[float, str, dict[str, float | int], np.ndarray, np.ndarray] | None = None
    ridge_valid = fit_ridge(train, valid, 1.0)
    ridge_test = fit_ridge(train, test, 1.0)
    candidates: list[tuple[str, dict[str, float | int], np.ndarray, np.ndarray]] = [("ridge", {"alpha": 1.0}, ridge_valid, ridge_test)]
    for alpha in [0.003, 0.01, 0.03]:
        for gamma in [0.01, 0.03, 0.1]:
            pred_valid, pred_test = fit_krr(train, [valid, test], alpha, gamma)
            candidates.append(("rbf_krr", {"alpha": alpha, "gamma": gamma}, pred_valid, pred_test))
            train_pred = fit_krr(train, [train], alpha, gamma)[0]
            residual = train.y.to_numpy(dtype=np.float64) - train_pred
            for k in [20, 40, 80]:
                for scale in [0.75, 1.5, 3.0]:
                    avg_valid = local_average(residual, train, valid, k, scale)
                    avg_test = local_average(residual, train, test, k, scale)
                    for shrink in [0.25, 0.5, 0.75, 1.0]:
                        candidates.append(("density_aware_local_krr", {"alpha": alpha, "gamma": gamma, "k": k, "bandwidth_scale": scale, "shrink": shrink}, pred_valid + shrink * avg_valid, pred_test + shrink * avg_test))
    for name, params, pred_valid, pred_test in candidates:
        valid_stats, _ = error_stats(valid.y, pred_valid)
        score = valid_stats["mean_3d_error_mm"]
        if best is None or score < best[0]:
            best = (score, name, params, pred_valid, pred_test)
    assert best is not None
    _, selected_name, selected_params, selected_valid, selected_test = best
    for bundle, pred in [(train, fit_krr(train, [train], selected_params.get("alpha", 0.01), selected_params.get("gamma", 0.03))[0] if selected_name != "ridge" else fit_ridge(train, train, 1.0)), (valid, selected_valid), (test, selected_test)]:
        stats, err = error_stats(bundle.y, pred)
        metrics.append({"seed": seed, "split": bundle.name, "model_name": selected_name, "device_used": "cuda", "cuda_device_name": device_name, **selected_params, **stats})
    pred_frame = test.meta.copy()
    pred_frame["seed"] = seed
    pred_frame["model_name"] = selected_name
    _, test_err = error_stats(test.y, selected_test)
    pred_frame["error_3d_mm"] = test_err
    return metrics, pred_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default="20260601_2link_density_aware_local_krr_smoke")
    parser.add_argument("--seeds", default="0-2")
    args = parser.parse_args()
    device_name = require_cuda()
    df = pd.read_csv(args.dataset_root / "dataset.csv")
    features = read_lines(args.dataset_root / "feature_columns.txt")
    labels = read_lines(args.dataset_root / "label_columns.txt")
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, float | int | str]] = []
    all_preds: list[pd.DataFrame] = []
    for seed in parse_seeds(args.seeds):
        metrics, preds = evaluate_seed(df, features, labels, seed, device_name)
        all_metrics.extend(metrics)
        all_preds.append(preds)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(run_dir / "seed_split_metrics.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)
    test = metrics_df[metrics_df["split"] == "test"]
    summary = test.select_dtypes(include=[np.number]).mean().to_dict()
    summary.update({"run_name": args.run_name, "dataset_root": str(args.dataset_root), "features": features, "labels": labels, "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")[:2000], "created_at": datetime.now().isoformat(timespec="seconds")})
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(f"# 2-Link Density-aware Local KRR smoke\n\n- dataset: `{args.dataset_root}`\n- seeds: `{args.seeds}`\n- device: `{device_name}`\n- test mean 3D error: `{summary.get('mean_3d_error_mm'):.4f} mm`\n- test median: `{summary.get('median_3d_error_mm'):.4f} mm`\n- test p95: `{summary.get('p95_3d_error_mm'):.4f} mm`\n- pass@2mm: `{summary.get('pass_2mm_rate'):.4f}`\n", encoding="utf-8")
    print(run_dir)
    print(test[["seed", "model_name", "mean_3d_error_mm", "median_3d_error_mm", "p95_3d_error_mm", "pass_2mm_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
