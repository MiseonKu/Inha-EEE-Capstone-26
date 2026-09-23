"""
Optimize final motor angles using a trained end-effector prediction model.

This module is designed to sit after the baseline ML model in the pipeline:

    target end-effector position -> candidate motor angles
    -> predicted actual position from ML model -> objective minimization
    -> final motor angle command

Expected location:

    PROJECT_ROOT/scripts/control/motor_angle_optimizer.py

Default model path:

    PROJECT_ROOT/outputs/random_forest_v2/random_forest_model.pkl

Example:

    python scripts/control/motor_angle_optimizer.py ^
        --target-x-mm 120 --target-y-mm 30 --target-z-mm 80 ^
        --initial-q1-deg 30 --initial-q2-deg 20
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "random_forest_v2" / "random_forest_model.pkl"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_2link_v1"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "motor_angle_optimizer" / "optimized_motor_angle.json"

RANDOM_STATE = 42


DEFAULT_FEATURE_COLUMNS = [
    "feature_q1_cmd_rad",
    "feature_q2_cmd_rad",
    "feature_p_target_tracker_origin_x_mm",
    "feature_p_target_tracker_origin_y_mm",
    "feature_p_target_tracker_origin_z_mm",
    "feature_ik_q1_delta_rad",
    "feature_ik_q2_delta_rad",
    "feature_pan_target_deg",
    "feature_lift_target_deg",
]


@dataclass(frozen=True)
class MotorAngleResult:
    q1_rad: float
    q2_rad: float
    q1_deg: float
    q2_deg: float
    pred_x_mm: float
    pred_y_mm: float
    pred_z_mm: float
    target_x_mm: float
    target_y_mm: float
    target_z_mm: float
    error_x_mm: float
    error_y_mm: float
    error_z_mm: float
    position_error_mm: float
    objective_value: float
    success: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find final q1/q2 motor angles by minimizing ML-predicted end-effector position error."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--target-x-mm", type=float, required=True)
    parser.add_argument("--target-y-mm", type=float, required=True)
    parser.add_argument("--target-z-mm", type=float, required=True)

    parser.add_argument("--initial-q1-deg", type=float, required=True)
    parser.add_argument("--initial-q2-deg", type=float, required=True)

    parser.add_argument("--q1-encoder-offset-deg", type=float, default=0.0)
    parser.add_argument("--q2-encoder-offset-deg", type=float, default=0.0)

    parser.add_argument("--q1-min-deg", type=float, default=-90.0)
    parser.add_argument("--q1-max-deg", type=float, default=90.0)
    parser.add_argument("--q2-min-deg", type=float, default=-45.0)
    parser.add_argument("--q2-max-deg", type=float, default=90.0)

    parser.add_argument("--angle-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--previous-q1-deg", type=float, default=None)
    parser.add_argument("--previous-q2-deg", type=float, default=None)

    parser.add_argument("--global-max-iter", type=int, default=120)
    parser.add_argument("--global-pop-size", type=int, default=15)
    parser.add_argument("--no-global-search", action="store_true")

    return parser.parse_args()


def read_column_file(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback

    columns = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return columns if columns else fallback


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found: {model_path}\n"
            "Run the RandomForest baseline first or pass --model-path."
        )

    with model_path.open("rb") as f:
        return pickle.load(f)


def load_feature_columns(dataset_root: Path) -> list[str]:
    feature_columns_path = dataset_root / "feature_columns.txt"
    return read_column_file(feature_columns_path, DEFAULT_FEATURE_COLUMNS)


def make_feature_row(
    feature_columns: list[str],
    q_rad: np.ndarray,
    target_xyz_mm: np.ndarray,
    initial_q_rad: np.ndarray,
    encoder_offsets_deg: np.ndarray,
) -> pd.DataFrame:
    q1_rad, q2_rad = q_rad
    initial_q1_rad, initial_q2_rad = initial_q_rad
    q1_deg, q2_deg = np.rad2deg(q_rad)
    initial_q1_deg, initial_q2_deg = np.rad2deg(initial_q_rad)
    q1_encoder_deg, q2_encoder_deg = np.rad2deg(q_rad) + encoder_offsets_deg
    target_x_mm, target_y_mm, target_z_mm = target_xyz_mm

    values = {
        "feature_q1_cmd_rad": q1_rad,
        "feature_q2_cmd_rad": q2_rad,
        "feature_p_target_tracker_origin_x_mm": target_x_mm,
        "feature_p_target_tracker_origin_y_mm": target_y_mm,
        "feature_p_target_tracker_origin_z_mm": target_z_mm,
        "feature_ik_q1_delta_rad": q1_rad - initial_q1_rad,
        "feature_ik_q2_delta_rad": q2_rad - initial_q2_rad,
        "feature_pan_target_deg": q1_deg,
        "feature_lift_target_deg": q2_deg,

        "q1_cmd_deg": q1_deg,
        "q2_cmd_deg": q2_deg,
        "q1_cmd_rad": q1_rad,
        "q2_cmd_rad": q2_rad,
        "q1_encoder_deg": q1_encoder_deg,
        "q2_encoder_deg": q2_encoder_deg,
        "p_target_x_mm": target_x_mm,
        "p_target_y_mm": target_y_mm,
        "p_target_z_mm": target_z_mm,
        "target_x_mm": target_x_mm,
        "target_y_mm": target_y_mm,
        "target_z_mm": target_z_mm,
        "ik_q1_delta_deg": q1_deg - initial_q1_deg,
        "ik_q2_delta_deg": q2_deg - initial_q2_deg,
        "ik_q1_delta_rad": q1_rad - initial_q1_rad,
        "ik_q2_delta_rad": q2_rad - initial_q2_rad,
        "pan_target_deg": q1_deg,
        "lift_target_deg": q2_deg,
    }

    missing = [column for column in feature_columns if column not in values]
    if missing:
        raise ValueError(
            "Cannot build optimizer input because these training features are unknown: "
            f"{missing}\n"
            "Add a mapping for them in make_feature_row()."
        )

    row = {column: values[column] for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def predict_position_mm(
    model: Any,
    feature_columns: list[str],
    q_rad: np.ndarray,
    target_xyz_mm: np.ndarray,
    initial_q_rad: np.ndarray,
    encoder_offsets_deg: np.ndarray,
) -> np.ndarray:
    X = make_feature_row(
        feature_columns,
        q_rad,
        target_xyz_mm,
        initial_q_rad,
        encoder_offsets_deg,
    )
    return np.asarray(model.predict(X)[0], dtype=float)


def objective(
    q_rad: np.ndarray,
    model: Any,
    feature_columns: list[str],
    target_xyz_mm: np.ndarray,
    initial_q_rad: np.ndarray,
    encoder_offsets_deg: np.ndarray,
    previous_q_rad: np.ndarray | None,
    angle_weight: float,
    smoothness_weight: float,
) -> float:
    pred_xyz_mm = predict_position_mm(
        model=model,
        feature_columns=feature_columns,
        q_rad=q_rad,
        target_xyz_mm=target_xyz_mm,
        initial_q_rad=initial_q_rad,
        encoder_offsets_deg=encoder_offsets_deg,
    )

    position_error_mm = np.linalg.norm(pred_xyz_mm - target_xyz_mm)
    initial_angle_penalty = np.linalg.norm(q_rad - initial_q_rad)
    cost = position_error_mm + angle_weight * initial_angle_penalty

    if previous_q_rad is not None and smoothness_weight > 0:
        cost += smoothness_weight * np.linalg.norm(q_rad - previous_q_rad)

    return float(cost)


def optimize_motor_angle(
    model: Any,
    feature_columns: list[str],
    target_xyz_mm: np.ndarray,
    initial_q_rad: np.ndarray,
    encoder_offsets_deg: np.ndarray,
    q_bounds_rad: list[tuple[float, float]],
    previous_q_rad: np.ndarray | None = None,
    angle_weight: float = 0.05,
    smoothness_weight: float = 0.0,
    use_global_search: bool = True,
    global_max_iter: int = 120,
    global_pop_size: int = 15,
) -> MotorAngleResult:
    objective_fn = lambda q: objective(
        q_rad=np.asarray(q, dtype=float),
        model=model,
        feature_columns=feature_columns,
        target_xyz_mm=target_xyz_mm,
        initial_q_rad=initial_q_rad,
        encoder_offsets_deg=encoder_offsets_deg,
        previous_q_rad=previous_q_rad,
        angle_weight=angle_weight,
        smoothness_weight=smoothness_weight,
    )

    if use_global_search:
        global_result = differential_evolution(
            func=objective_fn,
            bounds=q_bounds_rad,
            maxiter=global_max_iter,
            popsize=global_pop_size,
            tol=1e-4,
            seed=RANDOM_STATE,
            polish=False,
            updating="deferred",
            workers=1,
        )
        start_q_rad = global_result.x
    else:
        start_q_rad = initial_q_rad

    local_result = minimize(
        fun=objective_fn,
        x0=start_q_rad,
        bounds=q_bounds_rad,
        method="L-BFGS-B",
    )

    q_opt_rad = np.asarray(local_result.x, dtype=float)
    pred_xyz_mm = predict_position_mm(
        model=model,
        feature_columns=feature_columns,
        q_rad=q_opt_rad,
        target_xyz_mm=target_xyz_mm,
        initial_q_rad=initial_q_rad,
        encoder_offsets_deg=encoder_offsets_deg,
    )

    error_xyz_mm = pred_xyz_mm - target_xyz_mm
    position_error_mm = np.linalg.norm(error_xyz_mm)

    return MotorAngleResult(
        q1_rad=float(q_opt_rad[0]),
        q2_rad=float(q_opt_rad[1]),
        q1_deg=float(np.rad2deg(q_opt_rad[0])),
        q2_deg=float(np.rad2deg(q_opt_rad[1])),
        pred_x_mm=float(pred_xyz_mm[0]),
        pred_y_mm=float(pred_xyz_mm[1]),
        pred_z_mm=float(pred_xyz_mm[2]),
        target_x_mm=float(target_xyz_mm[0]),
        target_y_mm=float(target_xyz_mm[1]),
        target_z_mm=float(target_xyz_mm[2]),
        error_x_mm=float(error_xyz_mm[0]),
        error_y_mm=float(error_xyz_mm[1]),
        error_z_mm=float(error_xyz_mm[2]),
        position_error_mm=float(position_error_mm),
        objective_value=float(local_result.fun),
        success=bool(local_result.success),
        message=str(local_result.message),
    )


def save_result(result: MotorAngleResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    model_path = args.model_path.resolve()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output_path.resolve()

    target_xyz_mm = np.array(
        [args.target_x_mm, args.target_y_mm, args.target_z_mm],
        dtype=float,
    )
    initial_q_rad = np.deg2rad([args.initial_q1_deg, args.initial_q2_deg])
    encoder_offsets_deg = np.array(
        [args.q1_encoder_offset_deg, args.q2_encoder_offset_deg],
        dtype=float,
    )

    q_bounds_rad = [
        tuple(np.deg2rad([args.q1_min_deg, args.q1_max_deg])),
        tuple(np.deg2rad([args.q2_min_deg, args.q2_max_deg])),
    ]

    previous_q_rad = None
    if args.previous_q1_deg is not None and args.previous_q2_deg is not None:
        previous_q_rad = np.deg2rad([args.previous_q1_deg, args.previous_q2_deg])

    model = load_model(model_path)
    feature_columns = load_feature_columns(dataset_root)

    result = optimize_motor_angle(
        model=model,
        feature_columns=feature_columns,
        target_xyz_mm=target_xyz_mm,
        initial_q_rad=initial_q_rad,
        encoder_offsets_deg=encoder_offsets_deg,
        q_bounds_rad=q_bounds_rad,
        previous_q_rad=previous_q_rad,
        angle_weight=args.angle_weight,
        smoothness_weight=args.smoothness_weight,
        use_global_search=not args.no_global_search,
        global_max_iter=args.global_max_iter,
        global_pop_size=args.global_pop_size,
    )

    save_result(result, output_path)

    print("[FINAL MOTOR ANGLE]")
    print(f"q1: {result.q1_deg:.4f} deg ({result.q1_rad:.6f} rad)")
    print(f"q2: {result.q2_deg:.4f} deg ({result.q2_rad:.6f} rad)")
    print()
    print("[PREDICTED END-EFFECTOR POSITION]")
    print(f"x: {result.pred_x_mm:.4f} mm")
    print(f"y: {result.pred_y_mm:.4f} mm")
    print(f"z: {result.pred_z_mm:.4f} mm")
    print(f"3D position error: {result.position_error_mm:.4f} mm")
    print()
    print(f"[DONE] saved: {output_path}")


if __name__ == "__main__":
    main()