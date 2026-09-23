# Claim Boundary

This repository verifies the paper artifacts for settled-position prediction. It does not claim final closed-loop control performance.

## Supported Claims

- The 1-link result is a v1000 single-protocol random-split prediction result.
- The final 1-link run reads `data/processed/clean_dataset_1link_v2/` with 881 rows and uses `v1000_only_random` to evaluate the 584 `test-260524-1` rows.
- The 2-link result is a random-split prediction result on `data/processed/clean_dataset_2link_v1/`, a 5,000-row synthetic-label 2-link package.
- The reported `pass@2mm` value is a position prediction metric, not closed-loop control success.
- Kabsch alignment is a coordinate-frame registration step for the 2-link dataset, not a physical robot-base calibration.
- The optimizer linkage smoke result shows that the learned forward model can be connected to an optimization objective.

## Explicit Non-Claims

- No final closed-loop controller is claimed.
- No RL compensation result is claimed; RL remains future work.
- The optimizer linkage smoke is not a final control-performance result.
- The included full processed datasets are enough to inspect row-level processed data and final-run evidence, but raw device-log replay is outside this repository.

## Dataset Caveats

- The 2-link manifest marks the dataset as synthetic through `synthetic_label=true`.
- The 2-link manifest records 5,000 clean rows; any paper/poster wording that implies 4,812 valid rows must be checked before public release.
- The 1-link calibration status is partial vertical-home based alignment, not full tracker-to-robot-base calibration.
