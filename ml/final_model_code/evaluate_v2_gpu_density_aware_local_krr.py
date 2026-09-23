from __future__ import annotations

"""GPU-only density-aware local correction for the v2 ARD KRR ensemble.

The previous best local blend used inverse-distance averaging over a fixed k.
This script broadens that idea by making each query's local bandwidth depend on
its kth-nearest-neighbor distance, then validation-selects either residual
correction or label blending. The goal is to reduce tail error without claiming
extrapolation robustness.
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
from evaluate_v2_gpu_ard_krr import load_data, profile_catalog
from evaluate_v2_gpu_ard_krr_ensemble import FIXED_CANDIDATES, combo_name, fit_candidate
from evaluate_v2_gpu_krr_robustness import DATASET_ROOT, split_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_gpu_density_aware_local_krr"


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--split-mode", default="v1000_only_random")
    parser.add_argument("--ensemble-selection-metric", choices=("mean", "p95", "mean_plus_0p1_p95"), default="mean")
    parser.add_argument("--correction-selection-metric", choices=("mean", "p95", "mean_plus_0p1_p95"), default="mean")
    parser.add_argument("--max-ensemble-size", type=int, default=8)
    parser.add_argument("--ks", default="5,10,15,20,30,40,60")
    parser.add_argument("--bandwidth-scales", default="0.5,0.75,1.0,1.5,2.0,3.0")
    parser.add_argument("--residual-shrinks", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--label-blends", default="0.1,0.2,0.35,0.5,0.75")
    return parser.parse_args()


def nvidia_smi_text() -> str:
    return subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")


def standard_feature_tensor(train_x: pd.DataFrame, query_x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    train_np = train_x.to_numpy(dtype=np.float64)
    query_np = query_x.to_numpy(dtype=np.float64)
    med = np.nanmedian(train_np, axis=0)
    train_np = np.where(np.isfinite(train_np), train_np, med)
    query_np = np.where(np.isfinite(query_np), query_np, med)
    mean = train_np.mean(axis=0)
    std = train_np.std(axis=0)
    std[std < 1e-12] = 1.0
    train_scaled = (train_np - mean) / std
    query_scaled = (query_np - mean) / std
    return (
        torch.as_tensor(train_scaled, dtype=torch.float64, device="cuda"),
        torch.as_tensor(query_scaled, dtype=torch.float64, device="cuda"),
    )


def adaptive_local_average(train_values: np.ndarray, train_x: pd.DataFrame, query_x: pd.DataFrame, k: int, bandwidth_scale: float) -> np.ndarray:
    train_t, query_t = standard_feature_tensor(train_x, query_x)
    dist = torch.cdist(query_t, train_t)
    k_eff = min(int(k), dist.shape[1])
    nn_dist, nn_idx = torch.topk(dist, k=k_eff, largest=False)
    kth = torch.clamp(nn_dist[:, -1:], min=1e-6)
    bandwidth = torch.clamp(kth * float(bandwidth_scale), min=1e-6)
    weights = torch.exp(-0.5 * (nn_dist / bandwidth) ** 2)
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
    values_t = torch.as_tensor(train_values, dtype=torch.float64, device="cuda")
    out = torch.sum(values_t[nn_idx] * weights.unsqueeze(-1), dim=1)
    return out.detach().cpu().numpy()


def fit_base_ensemble(train: dict[str, Any], valid: dict[str, Any], test: dict[str, Any], max_ensemble_size: int, selection_metric: str) -> dict[str, Any]:
    profiles = {profile["name"]: profile for profile in profile_catalog()}
    train_preds: list[np.ndarray] = []
    valid_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    for candidate in FIXED_CANDIDATES:
        model = fit_candidate(candidate, train, profiles)
        train_preds.append(model.predict(train["X"]))
        valid_preds.append(model.predict(valid["X"]))
        test_preds.append(model.predict(test["X"]))

    combos = [
        combo
        for size in range(1, min(max_ensemble_size, len(FIXED_CANDIDATES)) + 1)
        for combo in itertools.combinations(range(len(FIXED_CANDIDATES)), size)
    ]
    best: tuple[float, tuple[int, ...]] | None = None
    for combo in combos:
        pred_valid = np.mean([valid_preds[i] for i in combo], axis=0)
        stats, _ = error_stats(valid["y"], pred_valid)
        score = selection_score(stats, selection_metric)
        if best is None or score < best[0]:
            best = (score, combo)
    assert best is not None
    selected = best[1]
    return {
        "selected_combo": selected,
        "selected_combo_name": combo_name(selected),
        "train_pred": np.mean([train_preds[i] for i in selected], axis=0),
        "valid_pred": np.mean([valid_preds[i] for i in selected], axis=0),
        "test_pred": np.mean([test_preds[i] for i in selected], axis=0),
    }


def evaluate(
    df: pd.DataFrame,
    features: list[str],
    labels: list[str],
    seeds: list[int],
    split_mode: str,
    ensemble_selection_metric: str,
    correction_selection_metric: str,
    max_ensemble_size: int,
    ks: list[int],
    bandwidth_scales: list[float],
    residual_shrinks: list[float],
    label_blends: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    for seed in seeds:
        train, valid, test = split_data(df, features, labels, seed, split_mode)
        base = fit_base_ensemble(train, valid, test, max_ensemble_size, ensemble_selection_metric)
        train_y = train["y"].to_numpy(dtype=np.float64)
        residual_train = train_y - base["train_pred"]
        candidates: list[tuple[str, dict[str, Any], np.ndarray, np.ndarray]] = [
            ("base", {"k": 0, "bandwidth_scale": 0.0, "amount": 0.0}, base["valid_pred"], base["test_pred"])
        ]
        for k in ks:
            for bandwidth_scale in bandwidth_scales:
                avg_valid_res = adaptive_local_average(residual_train, train["X"], valid["X"], k, bandwidth_scale)
                avg_test_res = adaptive_local_average(residual_train, train["X"], test["X"], k, bandwidth_scale)
                avg_valid_label = adaptive_local_average(train_y, train["X"], valid["X"], k, bandwidth_scale)
                avg_test_label = adaptive_local_average(train_y, train["X"], test["X"], k, bandwidth_scale)
                for shrink in residual_shrinks:
                    candidates.append(
                        (
                            "adaptive_residual",
                            {"k": k, "bandwidth_scale": bandwidth_scale, "amount": shrink},
                            base["valid_pred"] + shrink * avg_valid_res,
                            base["test_pred"] + shrink * avg_test_res,
                        )
                    )
                for blend in label_blends:
                    candidates.append(
                        (
                            "adaptive_label_blend",
                            {"k": k, "bandwidth_scale": bandwidth_scale, "amount": blend},
                            (1.0 - blend) * base["valid_pred"] + blend * avg_valid_label,
                            (1.0 - blend) * base["test_pred"] + blend * avg_test_label,
                        )
                    )

        best: tuple[float, str, dict[str, Any], np.ndarray, np.ndarray] | None = None
        for method, params, pred_valid, pred_test in candidates:
            valid_stats, _ = error_stats(valid["y"], pred_valid)
            score = selection_score(valid_stats, correction_selection_metric)
            grid_rows.append(
                {
                    "split_mode": split_mode,
                    "seed": seed,
                    "selected_base_combo": base["selected_combo_name"],
                    "method": method,
                    **params,
                    "correction_selection_metric": correction_selection_metric,
                    "selection_score": score,
                    "device_used": "cuda",
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    **{f"valid_{k}": v for k, v in valid_stats.items()},
                }
            )
            if best is None or score < best[0]:
                best = (score, method, params, pred_valid, pred_test)
        assert best is not None
        _, method, params, pred_valid, pred_test = best
        split_preds = {"train": base["train_pred"], "valid": pred_valid, "test": pred_test}
        for bundle in (train, valid, test):
            pred = split_preds[bundle["split"]]
            stats, err = error_stats(bundle["y"], pred)
            metrics.append(
                {
                    "split_mode": split_mode,
                    "seed": seed,
                    "model_name": "cuda_density_aware_local_krr",
                    "split": bundle["split"],
                    "rows": len(bundle["X"]),
                    "selected_base_combo": base["selected_combo_name"],
                    "correction_method": method,
                    "selected_k": params["k"],
                    "selected_bandwidth_scale": params["bandwidth_scale"],
                    "selected_amount": params["amount"],
                    "device_used": "cuda",
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    **stats,
                }
            )
            if bundle["split"] == "test":
                p = bundle["meta"].reset_index(drop=True).copy()
                p["split_mode"] = split_mode
                p["seed"] = seed
                p["model_name"] = "cuda_density_aware_local_krr"
                p["correction_method"] = method
                p["selected_k"] = params["k"]
                p["selected_bandwidth_scale"] = params["bandwidth_scale"]
                p["selected_amount"] = params["amount"]
                p["error_3d_mm"] = err
                for i, label in enumerate(labels):
                    true = bundle["y"].reset_index(drop=True)[label].to_numpy()
                    p[f"true_{label}"] = true
                    p[f"pred_{label}"] = pred[:, i]
                    p[f"error_{label}"] = pred[:, i] - true
                pred_rows.append(p)
        torch.cuda.synchronize()
    return pd.DataFrame(metrics), pd.DataFrame(grid_rows), pd.concat(pred_rows, ignore_index=True)


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
    metrics, grid, preds = evaluate(
        df,
        features,
        labels,
        seeds,
        args.split_mode,
        args.ensemble_selection_metric,
        args.correction_selection_metric,
        args.max_ensemble_size,
        parse_ints(args.ks),
        parse_floats(args.bandwidth_scales),
        parse_floats(args.residual_shrinks),
        parse_floats(args.label_blends),
    )
    summary = summarize(metrics)
    metrics.to_csv(out / "seed_split_metrics.csv", index=False)
    grid.to_csv(out / "validation_density_grid.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)
    summary.to_csv(out / "repeated_summary.csv", index=False)
    manifest = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_status": gpu_status,
        "nvidia_smi": nvidia_smi_text(),
        "dataset": data_manifest,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "fixed_candidates": FIXED_CANDIDATES,
        "outputs": {
            "metrics": str(out / "seed_split_metrics.csv"),
            "validation_density_grid": str(out / "validation_density_grid.csv"),
            "predictions": str(out / "test_predictions.csv"),
            "summary": str(out / "repeated_summary.csv"),
        },
        "best_test_summary": summary.loc[summary["split"] == "test"].iloc[0].to_dict() if len(summary.loc[summary["split"] == "test"]) else {},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out), "gpu_status": gpu_status, "best_test_summary": manifest["best_test_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
