from __future__ import annotations

"""Smoke-test 2-Link optimizer linkage with the final local-kernel predictor.

This script does not claim physical execution performance. It checks whether the
2-Link forward model can be used inside an optimizer-style objective:

    fixed target xyz + candidate q command/encoder values
    -> predicted Vive xyz
    -> choose the candidate with minimum predicted target error

The candidate search is discrete over observed q-space rows, so the result is a
safe structural smoke test for the ML-to-optimizer connection.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from evaluate_2link_density_aware_local_krr import (
    SplitBundle,
    fit_krr,
    local_average,
    read_lines,
    require_cuda,
    split_frame,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_ROOT: Final[Path] = PROJECT_ROOT / "data" / "processed" / "clean_dataset_2link_v1"
OUTPUT_ROOT: Final[Path] = (
    PROJECT_ROOT
    / "outputs"
    / "by_date"
    / "2026-06-01-2link-v1-model-search"
    / "05_optimizer_linkage"
)
DEFAULT_Q_COLUMNS: Final[tuple[str, ...]] = (
    "q1_cmd_deg",
    "q2_cmd_deg",
    "q3_cmd_deg",
    "q1_encoder_deg",
    "q2_encoder_deg",
    "q3_encoder_deg",
)
DEFAULT_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "p_target_x_mm",
    "p_target_y_mm",
    "p_target_z_mm",
)


@dataclass(frozen=True, slots=True)
class LocalKernelParams:
    alpha: float
    gamma: float
    k: int
    bandwidth_scale: float
    shrink: float


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    seed: int
    targets: int
    candidates: int
    params: LocalKernelParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", default="20260601_2link_optimizer_linkage_smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=1500)
    return parser.parse_args()


def build_query_frame(
    target_row: pd.Series,
    candidates: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    query = candidates.loc[:, list(DEFAULT_Q_COLUMNS)].reset_index(drop=True).copy()
    for column in DEFAULT_TARGET_COLUMNS:
        query[column] = float(target_row[column])
    return query.loc[:, feature_columns]


def predict_density_aware(
    train: SplitBundle,
    query: SplitBundle,
    residual: np.ndarray,
    params: LocalKernelParams,
) -> np.ndarray:
    base_pred = fit_krr(train, [query], params.alpha, params.gamma)[0]
    local_pred = local_average(
        values=residual,
        train=train,
        query=query,
        k=params.k,
        scale=params.bandwidth_scale,
    )
    return base_pred + params.shrink * local_pred


def select_candidate_pool(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    if count >= len(df):
        return df.reset_index(drop=True)
    indices = rng.choice(len(df), size=count, replace=False)
    return df.iloc[np.sort(indices)].reset_index(drop=True)


def smoke_test(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    config: SmokeConfig,
) -> pd.DataFrame:
    train, _, test = split_frame(df, feature_columns, label_columns, config.seed)
    train_pred = fit_krr(train, [train], config.params.alpha, config.params.gamma)[0]
    residual = train.y.to_numpy(dtype=np.float64) - train_pred
    target_rows = test.x.join(test.y).head(config.targets).reset_index(drop=True)
    candidate_pool = select_candidate_pool(df, config.candidates, config.seed)
    records: list[dict[str, float | int | str]] = []

    for target_index, target_row in target_rows.iterrows():
        query_x = build_query_frame(target_row, candidate_pool, feature_columns)
        query = SplitBundle(
            name="candidate",
            x=query_x,
            y=pd.DataFrame(
                np.zeros((len(query_x), len(label_columns))),
                columns=label_columns,
            ),
            meta=pd.DataFrame(index=query_x.index),
        )
        pred = predict_density_aware(train, query, residual, config.params)
        target_xyz = target_row.loc[list(DEFAULT_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
        errors = np.linalg.norm(pred - target_xyz, axis=1)
        best_idx = int(np.argmin(errors))
        best_candidate = candidate_pool.iloc[best_idx]
        original_label = target_row.loc[label_columns].to_numpy(dtype=np.float64)
        original_target_error = float(np.linalg.norm(original_label - target_xyz))
        records.append(
            {
                "target_index": int(target_index),
                "candidate_count": int(len(candidate_pool)),
                "best_predicted_error_mm": float(errors[best_idx]),
                "original_label_target_error_mm": original_target_error,
                "best_q1_cmd_deg": float(best_candidate["q1_cmd_deg"]),
                "best_q2_cmd_deg": float(best_candidate["q2_cmd_deg"]),
                "best_q3_cmd_deg": float(best_candidate["q3_cmd_deg"]),
                "target_x_mm": float(target_xyz[0]),
                "target_y_mm": float(target_xyz[1]),
                "target_z_mm": float(target_xyz[2]),
                "pred_x_mm": float(pred[best_idx, 0]),
                "pred_y_mm": float(pred[best_idx, 1]),
                "pred_z_mm": float(pred[best_idx, 2]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    device_name = require_cuda()
    params = LocalKernelParams(alpha=0.003, gamma=0.1, k=20, bandwidth_scale=0.75, shrink=1.0)
    config = SmokeConfig(seed=args.seed, targets=args.targets, candidates=args.candidates, params=params)
    df = pd.read_csv(args.dataset_root / "dataset.csv")
    feature_columns = read_lines(args.dataset_root / "feature_columns.txt")
    label_columns = read_lines(args.dataset_root / "label_columns.txt")
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results = smoke_test(df, feature_columns, label_columns, config)
    results.to_csv(run_dir / "optimizer_linkage_smoke_results.csv", index=False)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device_name,
        "dataset_root": "data/processed/clean_dataset_2link_v1",
        "seed": config.seed,
        "targets": config.targets,
        "candidates": config.candidates,
        "params": asdict(config.params),
        "mean_best_predicted_error_mm": float(results["best_predicted_error_mm"].mean()),
        "median_best_predicted_error_mm": float(results["best_predicted_error_mm"].median()),
        "max_best_predicted_error_mm": float(results["best_predicted_error_mm"].max()),
        "pass_2mm_rate": float((results["best_predicted_error_mm"] <= 2.0).mean()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "# 2-Link optimizer linkage smoke\n\n"
        "This is a discrete candidate-search smoke test, not a physical execution claim.\n\n"
        "- dataset: `data/processed/clean_dataset_2link_v1`\n"
        f"- device: `{device_name}`\n"
        f"- targets: `{config.targets}`\n"
        f"- candidates per target: `{config.candidates}`\n"
        f"- mean predicted optimized error: `{summary['mean_best_predicted_error_mm']:.4f} mm`\n"
        f"- median predicted optimized error: `{summary['median_best_predicted_error_mm']:.4f} mm`\n"
        f"- max predicted optimized error: `{summary['max_best_predicted_error_mm']:.4f} mm`\n"
        f"- pass@2mm: `{summary['pass_2mm_rate']:.4f}`\n",
        encoding="utf-8",
    )
    print(run_dir)
    print(results[["target_index", "best_predicted_error_mm", "best_q1_cmd_deg", "best_q2_cmd_deg", "best_q3_cmd_deg"]].to_string(index=False))


if __name__ == "__main__":
    main()

