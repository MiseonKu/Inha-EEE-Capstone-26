from __future__ import annotations

import json
import shutil

import pandas as pd

from _twolink_common import *


def main() -> None:
    src_root = final_package_root() / "optimizer"
    for name in ["optimizer_linkage_smoke_results.csv", "optimizer_linkage_smoke_summary.json", "README.md"]:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, OPTIMIZER_AUDIT / name)
    results = pd.read_csv(OPTIMIZER_AUDIT / "optimizer_linkage_smoke_results.csv")
    summary = read_json(OPTIMIZER_AUDIT / "optimizer_linkage_smoke_summary.json")
    out = {
        "generated_at_utc": now(),
        "target_count": int(len(results)),
        "candidate_count_max": int(results["candidate_count"].max()) if "candidate_count" in results else None,
        "mean_best_predicted_error_mm": float(results["best_predicted_error_mm"].mean()),
        "median_best_predicted_error_mm": float(results["best_predicted_error_mm"].median()),
        "max_best_predicted_error_mm": float(results["best_predicted_error_mm"].max()),
        "pass_2mm_rate": float((results["best_predicted_error_mm"] <= 2.0).mean()),
        "summary_source": summary,
        "physical_claim_status": "STRUCTURAL_SMOKE_ONLY_NOT_PHYSICAL_EXECUTION",
    }
    write_json(OPTIMIZER_AUDIT / "optimizer_linkage_smoke_summary.json", out)
    lines = [
        "# 2-link Optimizer Linkage Audit",
        "",
        "This is discrete candidate-search smoke evidence only.",
        "",
        f"- target count: `{out['target_count']}`",
        f"- candidate count max: `{out['candidate_count_max']}`",
        f"- mean best predicted optimized error: `{out['mean_best_predicted_error_mm']:.4f} mm`",
        f"- median best predicted optimized error: `{out['median_best_predicted_error_mm']:.4f} mm`",
        f"- max best predicted optimized error: `{out['max_best_predicted_error_mm']:.4f} mm`",
        f"- pass@2mm: `{out['pass_2mm_rate']:.4f}`",
        "",
        "## Claim Boundary",
        "",
        "- selected `best_q1_cmd_deg`, `best_q2_cmd_deg`, `best_q3_cmd_deg` are candidate angles, not validated motor commands.",
        "- This is not physical execution.",
        "- This is not continuous optimization.",
        "- This is not final control performance.",
        "- Next step requires continuous optimization or simulation/physical validation.",
    ]
    (OPTIMIZER_AUDIT / "optimizer_linkage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **{k: out[k] for k in ["target_count", "pass_2mm_rate"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
