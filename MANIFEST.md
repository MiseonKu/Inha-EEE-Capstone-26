# Manifest And Linkage Map

This manifest explains how the submitted folders connect. `FILE_INDEX.txt` contains the complete file-by-file list generated from the actual package contents.

## 1. Experiment, Filter, And Hardware Preparation

- `experiment_preparation/fkik_workspace_point_algorithm/`: original capstone-code 03 FK/IK, workspace sampling, target generation, trial schedule, trajectory, tests, and docs.
- `experiment_preparation/p1_one_link_experiment/`: one-link experiment-preparation package from capstone-code 06, including config, kinematics, sampling, scripts, simulation notes, generated target/schedule outputs, and tests.
- `experiment_preparation/p2_two_link_experiment/`: two-link experiment-preparation package from capstone-code 06, including config, kinematics, sampling, trajectory, validation scripts, generated target/schedule/trajectory outputs, and tests.
- `filter_selection/`: MPU-6050 acquisition/calibration tools, 100 recorded straight-pose trials with five simultaneous filter outputs, filter-comparison results, and the Unit 1 IMU joint-angle estimator.
- `hardware/p1_one_link/SO101 Assembly-link11.step`: one-link hardware source geometry.
- `hardware/p2_two_link/2_LINK_SO101_ASSEMBLY.step`: two-link hardware source geometry.

These folders define the target space, geometry assumptions, IMU filtering evidence, and hardware source files used before ML preprocessing.

## 2. Data And Preprocessing

- `data/processed/clean_dataset_1link_v1000/`: final one-link v1000 processed dataset.
- `data/processed/clean_dataset_1link_v2/`: one-link merged clean dataset retained for context and final v1000 split provenance.
- `data/processed/clean_dataset_2link_v1/`: final two-link processed dataset.
- `data/source_processed/`: source-processed synthetic 2-link inputs used to back the clean 2-link dataset.
- `data/schema/`: data dictionary, units, and coordinate-frame notes.
- `data/build_clean_dataset_1link_v1000.py` and `data/build_clean_dataset_2link_v1.py`: runnable package-local preprocessing builders.
- `data/provenance_builders/`: calibration-gated source retained for provenance but not used as the final one-command rerun path.

The clean datasets are the direct input surface for the ML scripts.

## 3. ML Model Code And Evidence

- `ml/final_model_code/evaluate_v2_gpu_density_aware_local_krr.py`: final one-link Density-aware Local KRR evaluation script.
- `ml/final_model_code/evaluate_2link_density_aware_local_krr.py`: final two-link Density-aware Local KRR evaluation script.
- `ml/final_model_code/generate_2link_paper_figures.py`: final two-link figure generation script.
- `ml/final_model_code/optimizer_linkage_2link_smoke.py`: model-to-optimizer linkage smoke script.
- `ml/final_model_code/evaluate_v1000_gpu_kernel_ridge.py`, `evaluate_v2_gpu_ard_krr.py`, `evaluate_v2_gpu_ard_krr_ensemble.py`, `evaluate_v2_gpu_krr_robustness.py`: direct dependencies for the final one-link script.
- `ml/final_models/`: model cards and final-model documentation copied from the ML workspace.
- `ml/final_model_artifacts/`: fitted full-train joblib artifacts and model manifest.
- `ml/final_results/`: final run outputs, expected metrics, tables, and figures.

The ML code consumes `data/processed/` and produces the reported prediction metrics.

## 4. Optimizer Linkage

- `optimizer/source/`: sanitized 2-link optimizer-linkage source copied from the earlier reproduction package.
- `optimizer/results/`: optimizer linkage smoke outputs.
- `optimizer/figures/`: pipeline figure connecting ML prediction to optimizer usage.
- `optimizer/legacy_1link/`: legacy one-link optimizer reference retained only for history; it is not the final two-link control claim.

The optimizer material is included because the project used it as a downstream linkage demonstration. It does not establish closed-loop hardware performance.

## 5. Paper And Documentation

- `paper_reference/`: paper/poster PDFs and short README.
- `docs/`: data cards, model cards, claim boundary, source provenance, figures/tables notes, and artifact checklist.
- `environment/`: dependency specifications.
- `tools/verify_submission_package.py`: offline package verification script.

## 6. Exclusion Rationale

Raw tracker logs and interactive HTML analysis dumps are intentionally excluded because they are very large and contain local machine paths. The processed datasets, preprocessing scripts, manifests, schemas, checksums, and final run evidence are included instead, which is the cleaner reproduction surface for school submission.

## 7. Complete File List

See `FILE_INDEX.txt` for every submitted file path.
