# 1-Link v1000 Final Model

## Model

- Model family: `Density-aware Local KRR`
- Role: final 1-link v1000 settled-position forward model
- Input data: `data/processed/clean_dataset_1link_v1000/`
- Runnable script: `ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py`
- Fitted full-train artifact: `ml/final_model_artifacts/1link_density_aware_local_krr_fulltrain.joblib`
- Final 50-seed evidence: `ml/final_results/final_runs/1link_v1000_density_aware_local_krr_50seed/`

## Reported Metrics

| metric | value |
|---|---:|
| mean 3D error | 2.0135 mm |
| median 3D error | 1.5656 mm |
| p95 3D error | 5.0003 mm |
| pass@2mm | 62.68% |

## Package-Local Rerun Shape

From the submission package root, after activating the CUDA environment:

```powershell
python ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py --dataset-root data/processed/clean_dataset_1link_v1000 --seeds 0-49
```

CUDA is required for final metric reproduction. CPU-only execution is diagnostic only.

## Claim Boundary

This is a forward-model prediction result against Vive Tracker settled-position labels. It is not a closed-loop control benchmark.
