# Final Model Index

This folder summarizes the final model choice used in the submission package.
The runnable scripts, fitted inference artifacts, and metric evidence are stored in the package-local `ml/` layout.

| Target | Final model | Runnable script | Fitted artifact | Final evidence |
|---|---|---|---|---|
| 1-link v1000 | Density-aware Local KRR | `ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py` | `ml/final_model_artifacts/1link_density_aware_local_krr_fulltrain.joblib` | `ml/final_results/final_runs/1link_v1000_density_aware_local_krr_50seed/` |
| 2-link v1 | Density-aware Local KRR | `ml/final_model_code/evaluate_2link_density_aware_local_krr.py` | `ml/final_model_artifacts/2link_density_aware_local_krr_fulltrain.joblib` | `ml/final_results/final_runs/2link_density_aware_local_krr_50seed/` |

The `.joblib` files are full-train inference artifacts for inspection/smoke prediction. The reported paper metrics come from the repeated random-split run evidence in `ml/final_results/final_runs/`.
