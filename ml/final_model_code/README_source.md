# Final Model Code

This folder contains the model scripts submitted for reproduction. It is intentionally smaller than the original ML workspace and keeps only the scripts needed to inspect or rerun the final 1-link/2-link model path plus direct comparison/support scripts.

## Runnable Final Scripts

| Script | Role |
|---|---|
| `evaluate_v2_gpu_density_aware_local_krr.py` | Final 1-link Density-aware Local KRR evaluation. |
| `evaluate_2link_density_aware_local_krr.py` | Final 2-link Density-aware Local KRR evaluation. |
| `generate_2link_paper_figures.py` | Rebuilds 2-link paper/result figures from package-local outputs. |
| `optimizer_linkage_2link_smoke.py` | Checks linkage between prediction output and downstream optimizer source. |

## Supporting Scripts

| Script | Role |
|---|---|
| `evaluate_v1000_gpu_kernel_ridge.py` | GPU KRR baseline and shared 1-link utilities. |
| `evaluate_v2_gpu_ard_krr.py` | ARD KRR comparison candidate. |
| `evaluate_v2_gpu_ard_krr_ensemble.py` | ARD ensemble comparison candidate. |
| `evaluate_v2_gpu_krr_robustness.py` | 1-link robustness/split diagnostic. |

## Execution Rules

Run commands from the submission package root after activating the CUDA environment in `environment/`.

```powershell
python ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py --dataset-root data/processed/clean_dataset_1link_v1000 --seeds 0-49
python ml/final_model_code/evaluate_2link_density_aware_local_krr.py --dataset-root data/processed/clean_dataset_2link_v1 --seeds 0-49
```

The GPU evaluators require CUDA. The submission copy is archive-safe: if `.git` metadata is absent, run manifests record `archive_no_git_metadata` instead of failing.
