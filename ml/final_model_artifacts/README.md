# Model Artifacts

This folder contains fitted model artifacts for convenience when inspecting the paper package.

These files are **full-train inference artifact** exports. They are useful for loading the fitted forward-model behavior and running a smoke prediction from the bundled sample rows.

They are **not the original 50-seed evaluation object** used to produce the reported paper metrics. The reported metrics remain tied to the repeated random-split evaluation scripts and output tables documented in `docs/model_card.md` and `ml/final_results/expected/expected_metrics.json`.

## Included Files

| file | role |
|---|---|
| `1link_density_aware_local_krr_fulltrain.joblib` | 1-link full-dataset RBF KRR plus density-aware residual correction |
| `2link_density_aware_local_krr_fulltrain.joblib` | 2-link full-dataset RBF KRR plus density-aware residual correction |
| `model_manifest.json` | artifact metadata, row counts, schema, hyperparameters, and checksums |

## Prediction Interface

Each `.joblib` file stores a Python dictionary with:

- `feature_columns`: required input column order
- `label_columns`: predicted output column order
- `preprocess`: median fill, mean, and standard deviation arrays
- `model`: RBF KRR and local residual-correction parameters
- `arrays`: training feature matrix, dual coefficients, and training residuals

Run `python scripts/verify_package.py` from the repository root to load both artifacts and verify a sample-row prediction smoke test.
