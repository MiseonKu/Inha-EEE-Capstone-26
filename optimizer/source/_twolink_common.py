from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOS = ROOT / "source_repos"
ML_REPO = SOURCE_REPOS / "ML"
CAPSTONE_REPO = SOURCE_REPOS / "capstone-code"
SNAPSHOT = ROOT / "source_data_snapshot"
FINAL_SNAPSHOT = SNAPSHOT / "final_2link_package"
RAW_SNAPSHOT = SNAPSHOT / "raw_2link_source"
PROCESSED_SNAPSHOT = SNAPSHOT / "processed_2link_v1"
INVENTORY = SNAPSHOT / "repo_file_inventory"
DATA_AUDIT = ROOT / "data_audit"
PREDICTIONS = ROOT / "predictions"
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
OPTIMIZER_AUDIT = ROOT / "optimizer_audit"
REPORT = ROOT / "report"
FINAL_OFFICIAL_PREDICTIONS = PREDICTIONS / "final_official_by_date"
LOCAL_DIAGNOSTIC_PREDICTIONS = PREDICTIONS / "local_reimplementation_diagnostic"
FINAL_OFFICIAL_METRICS = METRICS / "final_official"
LOCAL_DIAGNOSTIC_METRICS = METRICS / "local_reimplementation_diagnostic"
FINAL_OFFICIAL_FIGURES = FIGURES / "final_official"
LOCAL_DIAGNOSTIC_FIGURES = FIGURES / "local_reimplementation_diagnostic"

RAW_REL = Path("data/2link_test-260525-1")
PROCESSED_REL = Path("data/processed/clean_dataset_2link_v1")
FINAL_CANDIDATES = [
    Path("outputs/final/final_meeting_2link_v1_20260601"),
    Path("outputs/final/final_2link_v1_20260601"),
]
FEATURES = [
    "q1_cmd_deg",
    "q2_cmd_deg",
    "q3_cmd_deg",
    "q1_encoder_deg",
    "q2_encoder_deg",
    "q3_encoder_deg",
    "p_target_x_mm",
    "p_target_y_mm",
    "p_target_z_mm",
]
LABELS = ["p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"]
THRESHOLDS = [1.0, 2.0, 2.5, 5.0, 10.0]
KNN_K = 20


def ensure_dirs() -> None:
    for path in [
        SOURCE_REPOS,
        FINAL_SNAPSHOT,
        RAW_SNAPSHOT,
        PROCESSED_SNAPSHOT,
        INVENTORY,
        DATA_AUDIT,
        PREDICTIONS,
        METRICS,
        FIGURES,
        FINAL_OFFICIAL_PREDICTIONS,
        LOCAL_DIAGNOSTIC_PREDICTIONS,
        FINAL_OFFICIAL_METRICS,
        LOCAL_DIAGNOSTIC_METRICS,
        FINAL_OFFICIAL_FIGURES,
        LOCAL_DIAGNOSTIC_FIGURES,
        OPTIMIZER_AUDIT,
        REPORT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def copy_contents(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)
    return True


def final_package_rel() -> Path | None:
    for rel in FINAL_CANDIDATES:
        if (ML_REPO / rel).exists():
            return rel
    return None


def final_package_root() -> Path:
    if (FINAL_SNAPSHOT / "README.md").exists():
        return FINAL_SNAPSHOT
    rel = final_package_rel()
    if rel is not None:
        return ML_REPO / rel
    raise FileNotFoundError("2-link final package was not found")


def processed_root() -> Path:
    if (PROCESSED_SNAPSHOT / "dataset.csv").exists():
        return PROCESSED_SNAPSHOT
    if (ML_REPO / PROCESSED_REL / "dataset.csv").exists():
        return ML_REPO / PROCESSED_REL
    raise FileNotFoundError("clean_dataset_2link_v1 was not found")


def load_dataset() -> tuple[pd.DataFrame, list[str], list[str], Path]:
    root = processed_root()
    df = pd.read_csv(root / "dataset.csv").reset_index(drop=False).rename(columns={"index": "row_index"})
    features = read_lines(root / "feature_columns.txt") or FEATURES
    labels = read_lines(root / "label_columns.txt") or LABELS
    return df, features, labels, root


def engineer(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df[features].astype(float).copy()
    for col in ["q1_cmd_deg", "q2_cmd_deg", "q3_cmd_deg", "q1_encoder_deg", "q2_encoder_deg", "q3_encoder_deg"]:
        if col in out.columns:
            rad = np.deg2rad(out[col].to_numpy(dtype=float))
            out[f"{col}_sin"] = np.sin(rad)
            out[f"{col}_cos"] = np.cos(rad)
    for idx in [1, 2, 3]:
        c = f"q{idx}_cmd_deg"
        e = f"q{idx}_encoder_deg"
        if c in out.columns and e in out.columns:
            out[f"q{idx}_cmd_minus_encoder"] = out[c] - out[e]
    out["target_r_mm"] = np.linalg.norm(out[["p_target_x_mm", "p_target_y_mm", "p_target_z_mm"]].to_numpy(dtype=float), axis=1)
    return out


def split_70_15_15(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return idx[:train_end], idx[train_end:valid_end], idx[valid_end:]


def standardize(train: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = train.to_numpy(dtype=float)
    x_query = query.to_numpy(dtype=float)
    median = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_query = np.where(np.isfinite(x_query), x_query, median)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x_train - mean) / std, (x_query - mean) / std, median, mean, std


def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T), 0.0)


def fit_landmark_krr(train_x: pd.DataFrame, train_y: pd.DataFrame, seed: int, landmarks: int = 600, alpha: float = 0.003, gamma: float = 0.1) -> dict[str, Any]:
    x_train, _, _, _, _ = standardize(train_x, train_x)
    rng = np.random.RandomState(7000 + seed)
    m = min(landmarks, len(x_train))
    centers_idx = rng.choice(len(x_train), size=m, replace=False)
    centers = x_train[centers_idx]
    phi = np.exp(-gamma * sqdist(x_train, centers))
    lhs = phi.T @ phi
    lhs.flat[:: lhs.shape[0] + 1] += alpha
    coef = np.linalg.solve(lhs, phi.T @ train_y.to_numpy(dtype=float))
    return {"train_x": train_x.copy(), "centers": centers, "coef": coef, "gamma": gamma, "seed": seed}


def predict_landmark(model: dict[str, Any], query_x: pd.DataFrame) -> np.ndarray:
    train_s, query_s, _, _, _ = standardize(model["train_x"], query_x)
    phi = np.exp(-float(model["gamma"]) * sqdist(query_s, model["centers"]))
    return phi @ model["coef"]


def local_average(values: np.ndarray, train_x: pd.DataFrame, query_x: pd.DataFrame, k: int = KNN_K, scale: float = 0.75) -> tuple[np.ndarray, pd.DataFrame]:
    train_s, query_s, _, _, _ = standardize(train_x, query_x)
    dist = np.sqrt(sqdist(query_s, train_s))
    k_eff = min(k, dist.shape[1])
    idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn = np.take_along_axis(dist, idx, axis=1)
    order = np.argsort(nn, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    bandwidth = np.maximum(nn[:, -1:] * scale, 1e-6)
    weights = np.exp(-0.5 * (nn / bandwidth) ** 2)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    avg = np.sum(values[idx] * weights[:, :, None], axis=1)
    rel = pd.DataFrame({
        "nn_distance": nn[:, 0],
        "knn_mean_distance": nn.mean(axis=1),
        "local_density": 1.0 / (1e-9 + nn.mean(axis=1)),
        "local_neighbor_count": k_eff,
        "ood_score": nn.mean(axis=1),
    })
    return avg, rel


def regression_stats(df: pd.DataFrame) -> dict[str, float]:
    err = df["error_3d_mm"].to_numpy(dtype=float)
    ex = df["error_x_mm"].to_numpy(dtype=float)
    ey = df["error_y_mm"].to_numpy(dtype=float)
    ez = df["error_z_mm"].to_numpy(dtype=float)
    return {
        "rows": int(len(df)),
        "mean_3d_error_mm": float(np.mean(err)),
        "median_3d_error_mm": float(np.median(err)),
        "rmse_3d_error_mm": float(math.sqrt(np.mean(err**2))),
        "p90_3d_error_mm": float(np.percentile(err, 90)),
        "p95_3d_error_mm": float(np.percentile(err, 95)),
        "max_3d_error_mm": float(np.max(err)),
        "x_mae_mm": float(np.mean(np.abs(ex))),
        "y_mae_mm": float(np.mean(np.abs(ey))),
        "z_mae_mm": float(np.mean(np.abs(ez))),
        "x_rmse_mm": float(math.sqrt(np.mean(ex**2))),
        "y_rmse_mm": float(math.sqrt(np.mean(ey**2))),
        "z_rmse_mm": float(math.sqrt(np.mean(ez**2))),
    }


ensure_dirs()
