# 1-Link v1000 Density-Aware Local KRR 50-Seed Run

- Script: `ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py`
- Dataset: `data/processed/clean_dataset_1link_v2/`
- Split mode: `v1000_only_random`
- Seeds: `0-49`
- Test mean 3D error: `2.0135 mm`
- Test median 3D error: `1.5656 mm`
- Test p95 3D error: `5.0003 mm`
- Pass@2mm: `0.6268`

`run_manifest.json` is sanitized to keep only repository-relative paths and reproducibility metadata.

`candidate_comparison.csv` and `figures/v1000_final_50seed_candidate_comparison.png` preserve the final 1-link model-candidate selection evidence.
