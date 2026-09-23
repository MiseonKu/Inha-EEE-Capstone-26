# Capstone Reproduction Submission Package

This folder is the local school-submission archive for reproducing the capstone result. It keeps the source code, processed datasets, model artifacts, optimizer linkage code, experiment-preparation code, and hardware fabrication source files together so another student or reviewer can inspect and rerun the development chain without relying on repository history or private local settings.

## Reproduction Scope

The reported result is a Vive Tracker-labeled position-prediction result for flexible-link SO-101-based manipulators. The model predicts the settled 3D end-effector position from command angles, encoder angles, and target positions. It is a forward-model / position-prediction package; it is not closed-loop control and not a closed-loop control benchmark.

Included:

- processed 1-link and 2-link datasets used by the final results
- runnable preprocessing/build scripts for the final 1-link v1000 and 2-link datasets, plus provenance-only calibration-gated source for 1-link v2
- final Density-aware Local KRR model code and directly required dependency scripts
- fitted full-train model artifacts and final run evidence
- optimizer linkage source and smoke-check outputs
- MPU-6050 data acquisition, attitude-filter comparison results, and 1-unit joint-angle estimation code
- capstone-code 03 workspace/FK-IK/target-generation source
- capstone-code 06 P1/P2 experiment-preparation source
- STEP hardware fabrication/source geometry files
- paper/poster reference PDFs and result figures/tables

Excluded:

- `.git` history and repository settings
- local agent/automation caches such as `.omx`, `.hermes`, `.codegraph`, `__pycache__`
- raw tracker logs and large interactive HTML dumps containing local machine paths
- private experiment-tracking endpoint/API/credential material
- scripts that only automate publishing, repository maintenance, or local agent workflows

## Folder Map

| Folder | Role |
|---|---|
| `data/` | Processed datasets, source-processed 2-link synthetic inputs, schemas, checksums, and preprocessing builders. |
| `ml/` | Final model code, final model cards, full-train fitted artifacts, final 50-seed run evidence, result tables, and figures. |
| `optimizer/` | Optimizer linkage source and smoke-check results. This demonstrates source linkage, not final robot-control performance. |
| `experiment_preparation/` | Target generation, schedule generation, FK/IK, trajectory, simulation/preparation code copied from capstone-code 03 and 06. |
| `filter_selection/` | MPU-6050 acquisition tools, attitude-filter comparison data/results, and Unit 1 IMU joint-angle estimation code. |
| `hardware/` | STEP files and hardware source notes for the one-link and two-link flexible-link setups. |
| `paper_reference/` | Paper/poster PDFs retained as reference material for the submission. |
| `docs/` | Data/model/claim-boundary documentation from the previous reproduction package. |
| `environment/` | Python dependency files copied from the reproduction package. |
| `tools/` | Offline package verification script. |
| `FILE_INDEX.txt` | Complete file list for this submitted package. |
| `MANIFEST.md` | Human-readable linkage map across the submitted folders. |

## Reproduction Flow

1. Generate target/workspace/schedule material from `experiment_preparation/`.
2. Collect or simulate settled target samples according to the experiment-preparation docs.
3. Build clean datasets with the scripts in `data/`.
4. Train/evaluate final forward models with `ml/final_model_code/`.
5. Inspect final model artifacts and final run metrics in `ml/final_model_artifacts/` and `ml/final_results/`.
6. Inspect optimizer linkage in `optimizer/` to see how model predictions connect to the angle-search stage.
7. Use `hardware/` STEP sources to review the physical link geometry used by the experiments.

## Main Data Artifacts

| Dataset | Path | Role |
|---|---|---|
| 1-link v1000 | `data/processed/clean_dataset_1link_v1000/` | Final v1000 single-session subset, 584 clean rows. |
| 1-link v2 | `data/processed/clean_dataset_1link_v2/` | Full 1-link clean package used for final v1000 split/evidence context. |
| 2-link v1 | `data/processed/clean_dataset_2link_v1/` | Final 2-link Kabsch-aligned clean dataset, 5,000 rows. |

## Main Model Artifacts

| Artifact | Path |
|---|---|
| 1-link fitted model | `ml/final_model_artifacts/1link_density_aware_local_krr_fulltrain.joblib` |
| 2-link fitted model | `ml/final_model_artifacts/2link_density_aware_local_krr_fulltrain.joblib` |
| 1-link final run | `ml/final_results/final_runs/1link_v1000_density_aware_local_krr_50seed/` |
| 2-link final run | `ml/final_results/final_runs/2link_density_aware_local_krr_50seed/` |
| Expected metrics | `ml/final_results/expected/expected_metrics.json` |

## Core Commands

From this package root:

```powershell
python tools\verify_submission_package.py .
```

For full model reruns, create a Python environment from `environment/requirements.txt` or `environment/environment.yml`, then run the scripts under `ml/final_model_code/`. CUDA/GPU is expected for the final GPU KRR evaluation scripts. CPU-only reruns should be treated as diagnostic checks, not final performance evidence.

## Claim Boundary

The main reported metrics are settled-position prediction errors in millimeters. They compare predicted 3D end-effector positions with Vive Tracker-labeled settled positions. They do not prove real-time feedback control, closed-loop target tracking, or final hardware optimizer performance.
