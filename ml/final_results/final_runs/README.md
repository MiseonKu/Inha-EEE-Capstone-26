# Final Run Evidence

This directory contains privacy-sanitized final evaluation outputs for the reported paper metrics.

| Run | Dataset | Evidence |
|---|---|---|
| `1link_v1000_density_aware_local_krr_50seed/` | `data/processed/clean_dataset_1link_v2/` with `v1000_only_random` | 50-seed metrics, predictions, summary, validation-density grid, sanitized run manifest. |
| `2link_density_aware_local_krr_50seed/` | `data/processed/clean_dataset_2link_v1/` | 50-seed metrics, predictions, sanitized run manifest. |

The `.joblib` files under `ml/final_model_artifacts/` are full-train inference artifacts. They are not the original repeated-evaluation objects, so the metric provenance lives here.
