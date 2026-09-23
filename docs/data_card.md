# Data Card

## Scope

This repository provides the school-submission artifact package for paper verification and reproduction review. It includes full processed row-level datasets, selected source-processed 2-link synthetic CSV inputs, sample rows, manifests, checksums, and final run evidence.

Raw tracker/device logs and private machine-specific paths are not included.

## Included Data

- `data/processed/clean_dataset_1link_v2/`: 881 cleaned 1-link rows. The final 1-link run reads this package and applies `split_mode=v1000_only_random`.
- `data/processed/clean_dataset_1link_v1000/`: 584-row convenience v1000 subset from `test-260524-1`.
- `data/processed/clean_dataset_2link_v1/`: 5,000 cleaned 2-link synthetic-label rows.
- `data/source_processed/2link_synthetic/`: source-processed synthetic target plan and supervised samples used by the 2-link builder.
- `ml/final_results/final_runs/`: sanitized 50-seed metric, prediction, and run-manifest evidence.

## Excluded Data

- raw tracker logs
- full raw session folders
- private/local paths
- cache folders and temporary experiment dumps
- failed or non-final experiment outputs

## Dataset Notes

The 1-link reported metric is the v1000-only random-split result. The run manifest therefore records 881 available rows in `clean_dataset_1link_v2`, while the reported split selects the 584 `test-260524-1` rows.

The 2-link package contains 5,000 synthetic-label rows. It uses Kabsch registration into the target model frame, which is coordinate-frame registration evidence, not robot-base calibration.
