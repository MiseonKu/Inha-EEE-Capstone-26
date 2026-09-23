# Processed Datasets

This directory contains the full processed datasets included for the school-submission reproduction package.

The included datasets are derived, cleaned CSV artifacts. Raw tracker logs and temporary experiment dumps are still excluded.

| Dataset | Rows | Purpose |
|---|---:|---|
| `clean_dataset_1link_v2/` | 881 | Full 1-link clean package used by the final v1000-only random-split evaluation run. |
| `clean_dataset_1link_v1000/` | 584 | Convenience single-session v1000 subset from `test-260524-1`. |
| `clean_dataset_2link_v1/` | 5,000 | Final 2-link synthetic-label clean package with Kabsch-aligned target-frame labels. |

Each dataset directory includes:

- `dataset.csv`: row-level processed data.
- `manifest.json`: build, filtering, coordinate-frame, and row-count provenance.
- `session_summary.csv`: per-source session summary.
- `feature_columns.txt`: model input columns.
- `label_columns.txt`: model label columns.
