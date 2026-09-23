from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from evaluate_2link_density_aware_local_krr import (
    error_stats,
    fit_krr,
    local_average,
    read_lines,
    require_cuda,
    split_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "clean_dataset_2link_v1"
RUN_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "by_date"
    / "2026-06-01-2link-v1-model-search"
    / "02_local_kernel_candidates"
    / "20260601_2link_density_aware_local_krr_50seed"
)
BASELINE_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "by_date"
    / "2026-06-01-2link-v1-model-search"
    / "baseline20_to_density_aware50_summary.csv"
)
FINAL_ROOT = PROJECT_ROOT / "outputs" / "final" / "final_2link_v1_20260601"
FIGURE_ROOT = FINAL_ROOT / "figures"

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
FINAL_PARAMS = {"alpha": 0.003, "gamma": 0.1, "k": 20, "bandwidth_scale": 0.75, "shrink": 1.0}


def configure_plot_style() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = "Malgun Gothic"
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    out = FIGURE_ROOT / filename
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def reconstruct_seed0_predictions(df: pd.DataFrame) -> pd.DataFrame:
    train, _valid, test = split_frame(df, FEATURES, LABELS, seed=0)
    train_pred = fit_krr(train, [train], FINAL_PARAMS["alpha"], FINAL_PARAMS["gamma"])[0]
    residual = train.y.to_numpy(dtype=np.float64) - train_pred
    base_test = fit_krr(train, [test], FINAL_PARAMS["alpha"], FINAL_PARAMS["gamma"])[0]
    local = local_average(
        residual,
        train,
        test,
        FINAL_PARAMS["k"],
        FINAL_PARAMS["bandwidth_scale"],
    )
    pred = base_test + FINAL_PARAMS["shrink"] * local
    stats, errors = error_stats(test.y, pred)
    frame = test.x.reset_index(drop=True).copy()
    true = test.y.reset_index(drop=True)
    for axis, label_col in zip(["x", "y", "z"], LABELS, strict=True):
        frame[f"true_{axis}_mm"] = true[label_col]
    for idx, axis in enumerate(["x", "y", "z"]):
        frame[f"pred_{axis}_mm"] = pred[:, idx]
        frame[f"residual_{axis}_mm"] = pred[:, idx] - frame[f"true_{axis}_mm"]
    frame["error_3d_mm"] = errors
    frame.attrs["stats"] = stats
    return frame


def plot_dataset_overview(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    target_cols = ["p_target_x_mm", "p_target_y_mm", "p_target_z_mm"]
    label_cols = ["p_vive_x_mm", "p_vive_y_mm", "p_vive_z_mm"]
    q_cols = ["q1_cmd_deg", "q2_cmd_deg", "q3_cmd_deg"]
    for ax, col in zip(axes[0], q_cols, strict=True):
        ax.hist(df[col], bins=36, color="#4C78A8", alpha=0.85)
        ax.set_title(f"{col} 분포")
        ax.set_xlabel("deg")
        ax.set_ylabel("row")
    for ax, target_col, label_col, axis_name in zip(axes[1], target_cols, label_cols, ["X", "Y", "Z"], strict=True):
        ax.hist(df[target_col], bins=40, alpha=0.65, label="target", color="#59A14F")
        ax.hist(df[label_col], bins=40, alpha=0.55, label="aligned label", color="#F28E2B")
        ax.set_title(f"{axis_name}축 target/label 분포")
        ax.set_xlabel("mm")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("2-Link v1 데이터셋 입력 각도 및 target/label 분포", fontsize=15)
    return save(fig, "2026-06-01_2link_dataset_overview.png")


def plot_target_workspace(df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(12, 5.4))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    sample = df.sample(n=min(1800, len(df)), random_state=42)
    ax3d.scatter(sample["p_target_x_mm"], sample["p_target_y_mm"], sample["p_target_z_mm"], s=5, alpha=0.45)
    ax3d.set_title("target workspace 3D 분포")
    ax3d.set_xlabel("X mm")
    ax3d.set_ylabel("Y mm")
    ax3d.set_zlabel("Z mm")
    ax2 = fig.add_subplot(1, 2, 2)
    sc = ax2.scatter(df["p_target_x_mm"], df["p_target_y_mm"], c=df["p_target_z_mm"], s=7, alpha=0.55, cmap="viridis")
    ax2.set_title("target XY projection (색: Z)")
    ax2.set_xlabel("target X mm")
    ax2.set_ylabel("target Y mm")
    fig.colorbar(sc, ax=ax2, label="target Z mm")
    fig.suptitle("2-Link v1 목표 위치 공간 커버리지", fontsize=15)
    return save(fig, "2026-06-01_2link_target_workspace.png")


def plot_alignment_residual(df: pd.DataFrame) -> Path:
    residual = pd.DataFrame(
        {
            "x": df["p_vive_x_mm"] - df["p_target_x_mm"],
            "y": df["p_vive_y_mm"] - df["p_target_y_mm"],
            "z": df["p_vive_z_mm"] - df["p_target_z_mm"],
        }
    )
    residual["3d"] = np.linalg.norm(residual[["x", "y", "z"]].to_numpy(), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].boxplot([residual[c] for c in ["x", "y", "z"]], tick_labels=["X", "Y", "Z"], showfliers=False)
    axes[0].set_title("Kabsch 정합 후 axis residual")
    axes[0].set_ylabel("label - target (mm)")
    axes[1].hist(residual["3d"], bins=45, color="#E15759", alpha=0.8)
    axes[1].axvline(residual["3d"].mean(), color="black", linestyle="--", label=f"mean={residual['3d'].mean():.1f}mm")
    axes[1].set_title("정합 후 target-label 3D residual")
    axes[1].set_xlabel("3D residual (mm)")
    axes[1].legend(frameon=False)
    fig.suptitle("2-Link synthetic label frame 정합 잔차: 모델이 보정해야 하는 비선형 차이", fontsize=14)
    return save(fig, "2026-06-01_2link_alignment_residual.png")


def plot_true_vs_pred(seed0: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, axis_name in zip(axes, ["x", "y", "z"], strict=True):
        true = seed0[f"true_{axis_name}_mm"]
        pred = seed0[f"pred_{axis_name}_mm"]
        ax.scatter(true, pred, s=10, alpha=0.55, color="#4C78A8")
        low = min(true.min(), pred.min())
        high = max(true.max(), pred.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1.1, linestyle="--")
        ax.set_title(f"{axis_name.upper()} axis")
        ax.set_xlabel("true mm")
        ax.set_ylabel("predicted mm")
    fig.suptitle("Seed 0 test split: 실제 label 대비 예측 좌표", fontsize=15)
    return save(fig, "2026-06-01_2link_seed0_true_vs_pred.png")


def plot_seed0_error_views(seed0: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(seed0["error_3d_mm"], bins=50, color="#4C78A8", alpha=0.85)
    axes[0].axvline(2.0, color="#E15759", linestyle="--", label="2mm 기준")
    axes[0].axvline(2.5, color="#F28E2B", linestyle="--", label="2.5mm 기준")
    axes[0].set_title("Seed 0 test 3D error histogram")
    axes[0].set_xlabel("3D error (mm)")
    axes[0].legend(frameon=False)
    sorted_err = np.sort(seed0["error_3d_mm"].to_numpy())
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, cdf, color="#59A14F", linewidth=2)
    axes[1].axvline(2.0, color="#E15759", linestyle="--")
    axes[1].axvline(2.5, color="#F28E2B", linestyle="--")
    axes[1].set_title("Seed 0 test cumulative distribution")
    axes[1].set_xlabel("3D error (mm)")
    axes[1].set_ylabel("cumulative rate")
    fig.suptitle("2-Link final model seed 0 오차 분포", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_seed0_error_hist_cdf.png"))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(seed0["p_target_x_mm"], seed0["p_target_y_mm"], c=seed0["error_3d_mm"], s=13, alpha=0.72, cmap="magma")
    ax.set_title("Seed 0 test target XY별 3D error")
    ax.set_xlabel("target X mm")
    ax.set_ylabel("target Y mm")
    fig.colorbar(sc, ax=ax, label="3D error (mm)")
    paths.append(save(fig, "2026-06-01_2link_seed0_error_by_target_xy.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, axis_name in zip(axes, ["x", "y", "z"], strict=True):
        ax.hist(seed0[f"residual_{axis_name}_mm"], bins=42, alpha=0.82, color="#9C755F")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"{axis_name.upper()} residual")
        ax.set_xlabel("pred - true (mm)")
    fig.suptitle("Seed 0 test axis residual 분포", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_seed0_axis_residuals.png"))
    return paths


def plot_50seed_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    test = metrics[metrics["split"] == "test"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    metric_cols = ["mean_3d_error_mm", "p95_3d_error_mm", "pass_2mm_rate"]
    titles = ["Mean 3D error", "P95 3D error", "Pass@2mm"]
    for ax, col, title in zip(axes, metric_cols, titles, strict=True):
        ax.boxplot(test[col], vert=True, widths=0.45, showfliers=True)
        jitter = np.random.default_rng(0).normal(1, 0.025, size=len(test))
        ax.scatter(jitter, test[col], s=13, alpha=0.65, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_ylabel(col)
    fig.suptitle("50개 random seed test split 안정성", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_50seed_stability_boxplot.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    pooled = predictions["error_3d_mm"].to_numpy(dtype=np.float64)
    axes[0].hist(pooled, bins=70, color="#4C78A8", alpha=0.82)
    axes[0].axvline(2.0, color="#E15759", linestyle="--", label="2mm")
    axes[0].axvline(2.5, color="#F28E2B", linestyle="--", label="2.5mm")
    axes[0].set_title("50seed pooled test error histogram")
    axes[0].set_xlabel("3D error (mm)")
    axes[0].legend(frameon=False)
    sorted_err = np.sort(pooled)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, cdf, color="#59A14F", linewidth=2)
    axes[1].axvline(2.0, color="#E15759", linestyle="--")
    axes[1].axvline(2.5, color="#F28E2B", linestyle="--")
    axes[1].set_title("50seed pooled CDF")
    axes[1].set_xlabel("3D error (mm)")
    axes[1].set_ylabel("cumulative rate")
    fig.suptitle("2-Link final model 50seed pooled tail 분포", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_50seed_error_hist_cdf.png"))
    return paths


def plot_model_comparison(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    names = summary["model_name"].astype(str).str.replace("_", " ").tolist()
    colors = ["#BAB0AC", "#F28E2B", "#4C78A8"][: len(summary)]
    for ax, col, title in zip(
        axes,
        ["mean_3d_error_mm", "p95_3d_error_mm", "pass_2mm_rate"],
        ["Mean 3D error (mm)", "P95 3D error (mm)", "Pass@2mm"],
        strict=True,
    ):
        ax.bar(names, summary[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for idx, value in enumerate(summary[col]):
            label = f"{value:.3f}" if col != "pass_2mm_rate" else f"{value:.3f}"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("2-Link 후보 모델 대비 최종 모델 개선", fontsize=15)
    return save(fig, "2026-06-01_2link_model_comparison_summary.png")


def plot_pipeline_diagrams() -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        ("2-Link 명령/엔코더\n+ 목표 좌표", 0.08),
        ("RBF KRR\n전역 비선형 mapping", 0.32),
        ("train residual\n근방 sample 탐색", 0.55),
        ("Local residual 보정\n밀도-aware 평균", 0.75),
        ("Tracker 좌표 예측\n3D error 평가", 0.92),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F7F7F7", "edgecolor": "#4C78A8"}, fontsize=11)
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:], strict=True):
        ax.annotate("", xy=(x2 - 0.07, 0.55), xytext=(x1 + 0.07, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"})
    ax.set_title("Density-aware Local KRR 구조: 전역 커널 회귀 + 국소 residual 보정", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_density_aware_local_krr_diagram.png"))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        ("목표 end-effector\n좌표 입력", 0.1),
        ("후보 motor angle\n또는 연속 최적화", 0.33),
        ("학습된 forward model\n위치 예측", 0.56),
        ("목적함수\n||pred-target||", 0.76),
        ("최종 motor angle\n산출", 0.92),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FBFBFB", "edgecolor": "#59A14F"}, fontsize=11)
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:], strict=True):
        ax.annotate("", xy=(x2 - 0.07, 0.55), xytext=(x1 + 0.07, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"})
    ax.text(0.5, 0.18, "현재 optimizer smoke는 구조 연결 검증이며, 실제 제어 성능 claim은 추가 검증 후 분리해서 주장", ha="center", fontsize=10)
    ax.set_title("ML forward model 이후 optimizer 연결 구조", fontsize=15)
    paths.append(save(fig, "2026-06-01_2link_ml_to_optimizer_pipeline.png"))
    return paths


def plot_1link_2link_context() -> Path:
    context = pd.DataFrame(
        [
            {"dataset": "1-Link v1000", "mean": 2.0135, "median": 1.5656, "p95": 5.0003, "pass2": 0.6268},
            {"dataset": "2-Link v1", "mean": 0.7101, "median": 0.5193, "p95": 1.8215, "pass2": 0.9600},
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    width = 0.26
    x = np.arange(len(context))
    for offset, col, label, color in [(-width, "mean", "Mean", "#4C78A8"), (0, "median", "Median", "#59A14F"), (width, "p95", "P95", "#F28E2B")]:
        axes[0].bar(x + offset, context[col], width=width, label=label, color=color)
    axes[0].set_xticks(x, context["dataset"])
    axes[0].set_ylabel("3D error (mm)")
    axes[0].set_title("1-Link/2-Link 최종 forward model 오차")
    axes[0].legend(frameon=False)
    axes[1].bar(context["dataset"], context["pass2"], color=["#BAB0AC", "#4C78A8"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Pass@2mm")
    axes[1].set_ylabel("rate")
    for idx, value in enumerate(context["pass2"]):
        axes[1].text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle("최종 실험 맥락: 1-Link에서 찾은 구조를 2-Link v1에 적용", fontsize=15)
    return save(fig, "2026-06-01_1link_vs_2link_final_context.png")


def make_contact_sheet(paths: list[Path]) -> Path:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((480, 300))
            canvas = Image.new("RGB", (500, 340), "white")
            canvas.paste(thumb, ((500 - thumb.width) // 2, 10))
            thumbs.append(canvas)
            labels.append(path.name)
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 500, rows * 340), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 500, (idx // cols) * 340))
    out = FIGURE_ROOT / "2026-06-01_2link_paper_figure_contact_sheet.png"
    sheet.save(out)
    return out


def write_index(paths: list[Path], device_name: str, seed0: pd.DataFrame, metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"]
    stats = {
        "mean": float(test["mean_3d_error_mm"].mean()),
        "median": float(test["median_3d_error_mm"].mean()),
        "p95": float(test["p95_3d_error_mm"].mean()),
        "pass2": float(test["pass_2mm_rate"].mean()),
        "seed0_mean": float(seed0.attrs["stats"]["mean_3d_error_mm"]),
        "seed0_p95": float(seed0.attrs["stats"]["p95_3d_error_mm"]),
    }
    descriptions = {
        "2026-06-01_2link_dataset_overview.png": "데이터셋의 q1/q2/q3 명령각 분포와 target/정합 label의 축별 분포. 데이터 수집·전처리 설명용.",
        "2026-06-01_2link_target_workspace.png": "2-Link target workspace 커버리지. 실험 목표 좌표가 어떤 공간을 덮는지 설명할 때 사용.",
        "2026-06-01_2link_alignment_residual.png": "Kabsch frame 정합 후에도 남는 target-label residual. forward model이 단순 frame 변환이 아니라 비선형 residual까지 학습한다는 근거.",
        "2026-06-01_2link_model_comparison_summary.png": "Ridge, RBF KRR, Density-aware Local KRR 성능 비교. 최종 모델 선택 근거.",
        "2026-06-01_2link_seed0_true_vs_pred.png": "대표 seed 0 test split에서 x/y/z 실제 좌표 대비 예측 좌표.",
        "2026-06-01_2link_seed0_error_hist_cdf.png": "대표 seed 0 test split의 3D error histogram/CDF.",
        "2026-06-01_2link_seed0_error_by_target_xy.png": "target XY 위치별 3D error heat scatter. 특정 공간에서 tail이 생기는지 확인.",
        "2026-06-01_2link_seed0_axis_residuals.png": "대표 seed 0의 축별 residual 분포. 어느 축이 오차에 기여하는지 설명.",
        "2026-06-01_2link_50seed_stability_boxplot.png": "50개 random seed의 mean/p95/pass@2mm 안정성. seed 운이 아닌지 보여주는 검증 figure.",
        "2026-06-01_2link_50seed_error_hist_cdf.png": "50seed pooled test error tail. 논문에서 평균뿐 아니라 tail까지 제시할 때 사용.",
        "2026-06-01_2link_density_aware_50seed_comparison.png": "기존 50seed 핵심 비교 figure. 최종 수치 요약용.",
        "2026-06-01_2link_density_aware_50seed_tail.png": "기존 50seed tail 진단 figure. tail/극단값 설명용.",
        "2026-06-01_2link_density_aware_local_krr_diagram.png": "최종 모델 구조 개념도.",
        "2026-06-01_2link_ml_to_optimizer_pipeline.png": "ML forward model 이후 optimizer 연결 구조.",
        "2026-06-01_1link_vs_2link_final_context.png": "1-Link 최종 구조를 2-Link v1에 확장 적용한 맥락 비교.",
        "2026-06-01_2link_paper_figure_contact_sheet.png": "생성된 figure를 한 장에서 빠르게 검토하기 위한 contact sheet.",
    }
    lines = [
        "# 2-Link v1 논문용 figure 색인",
        "",
        f"- 생성 장치: CUDA / {device_name}",
        f"- 최종 모델: Density-aware Local KRR (`alpha={FINAL_PARAMS['alpha']}`, `gamma={FINAL_PARAMS['gamma']}`, `k={FINAL_PARAMS['k']}`, `bandwidth_scale={FINAL_PARAMS['bandwidth_scale']}`, `shrink={FINAL_PARAMS['shrink']}`)",
        f"- 50seed 평균 성능: mean `{stats['mean']:.4f}mm`, median `{stats['median']:.4f}mm`, p95 `{stats['p95']:.4f}mm`, Pass@2mm `{stats['pass2']:.4f}`",
        f"- seed0 재구성 성능: mean `{stats['seed0_mean']:.4f}mm`, p95 `{stats['seed0_p95']:.4f}mm`",
        "",
        "## Figure 목록과 논문 내 용도",
        "",
    ]
    for idx, path in enumerate(paths, 1):
        lines.append(f"{idx}. `figures/{path.name}`")
        lines.append(f"   - {descriptions.get(path.name, '논문/미팅용 보조 figure.')}")
    lines.extend(
        [
            "",
            "## 사용 시 주의",
            "",
            "- 이 figure pack은 2-Link forward model random split 결과를 설명한다.",
            "- optimizer figure는 구조 연결 설명용이며, 실제 제어 성공률 claim은 별도 optimizer/실기 검증 이후에만 사용한다.",
            "- 데이터셋은 `data/processed/clean_dataset_2link_v1` 기준이며, 원본 synthetic label은 Kabsch 정합 후에도 비선형 residual이 남는다.",
        ]
    )
    (FIGURE_ROOT / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (FIGURE_ROOT / "figure_generation_manifest.json").write_text(
        json.dumps(
            {
                "device": device_name,
                "dataset": str(DATASET_ROOT),
                "run_root": str(RUN_ROOT),
                "final_params": FINAL_PARAMS,
                "summary": stats,
                "figures": [path.name for path in paths],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    configure_plot_style()
    device_name = require_cuda()
    df = pd.read_csv(DATASET_ROOT / "dataset.csv")
    features = read_lines(DATASET_ROOT / "feature_columns.txt")
    labels = read_lines(DATASET_ROOT / "label_columns.txt")
    if features != FEATURES or labels != LABELS:
        raise RuntimeError(f"Unexpected schema: features={features}, labels={labels}")
    metrics = pd.read_csv(RUN_ROOT / "seed_split_metrics.csv")
    predictions = pd.read_csv(RUN_ROOT / "test_predictions.csv")
    summary = pd.read_csv(BASELINE_SUMMARY)
    seed0 = reconstruct_seed0_predictions(df)

    paths: list[Path] = []
    paths.append(plot_dataset_overview(df))
    paths.append(plot_target_workspace(df))
    paths.append(plot_alignment_residual(df))
    paths.append(plot_model_comparison(summary))
    paths.append(plot_true_vs_pred(seed0))
    paths.extend(plot_seed0_error_views(seed0))
    paths.extend(plot_50seed_metrics(metrics, predictions))
    paths.extend(plot_pipeline_diagrams())
    paths.append(plot_1link_2link_context())
    generated_names = {path.name for path in paths}
    for existing in sorted(FIGURE_ROOT.glob("2026-06-01_2link_density_aware_50seed_*.png")):
        if existing.name not in generated_names:
            paths.append(existing)
    contact_sheet = make_contact_sheet(paths)
    paths.append(contact_sheet)
    write_index(paths, device_name, seed0, metrics)
    print(f"Generated {len(paths)} figures in {FIGURE_ROOT}")
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
