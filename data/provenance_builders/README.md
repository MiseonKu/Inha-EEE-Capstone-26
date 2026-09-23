# Provenance Builders

This folder keeps preprocessing source that documents earlier development steps but is not a one-command rerun surface inside this submission archive.

- `build_clean_dataset_1link_v2_calibration_gated.py`: calibration-gated 1-link v2 builder retained for source provenance. It requires the earlier v1 input dataset and a reviewed calibration specification that are not part of the final submission rerun path. The final submitted 1-link result is reproduced from the included v2 dataset by `../build_clean_dataset_1link_v1000.py`.

Runnable package-local builders are kept directly under `data/`:

- `../build_clean_dataset_1link_v1000.py`
- `../build_clean_dataset_2link_v1.py`
