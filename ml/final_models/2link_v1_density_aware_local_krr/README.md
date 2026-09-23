# 2-Link v1 Final Model

## Model

- Model family: `Density-aware Local KRR`
- Role: final 2-link settled-position forward model
- Input data: `data/processed/clean_dataset_2link_v1/`
- Runnable script: `ml/final_model_code/evaluate_2link_density_aware_local_krr.py`
- Figure script: `ml/final_model_code/generate_2link_paper_figures.py`
- Optimizer smoke script: `ml/final_model_code/optimizer_linkage_2link_smoke.py`
- Fitted full-train artifact: `ml/final_model_artifacts/2link_density_aware_local_krr_fulltrain.joblib`
- Final 50-seed evidence: `ml/final_results/final_runs/2link_density_aware_local_krr_50seed/`

## Hyperparameters

| parameter | value |
|---|---:|
| alpha | 0.003 |
| gamma | 0.1 |
| k | 20 |
| bandwidth_scale | 0.75 |
| shrink | 1.0 |

## Reported Metrics

| metric | value |
|---|---:|
| mean 3D error | 0.7101 mm |
| median 3D error | 0.5193 mm |
| p95 3D error | 1.8215 mm |
| pass@2mm | 96.00% |

## Package-Local Rerun Shape

From the submission package root, after activating the CUDA environment:

```powershell
python ml/final_model_code/evaluate_2link_density_aware_local_krr.py --dataset-root data/processed/clean_dataset_2link_v1 --seeds 0-49 --run-name 20260601_2link_density_aware_local_krr_50seed
python ml/final_model_code/generate_2link_paper_figures.py
```

CUDA is required for final metric reproduction. CPU-only execution is diagnostic only.

## Claim Boundary

The metric claim is 2-link clean-dataset internal random-split forward prediction. It is not real hardware control performance. Optimizer material is included as source linkage and smoke evidence only.
