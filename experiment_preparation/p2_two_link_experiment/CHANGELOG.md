# Changelog

## 2026-05-23

### Added

- Created the P2 Windows/matplotlib experiment preparation package.
- Added authoritative P2 geometry:
  - `L1_effective = 299.6 mm`
  - `L2_effective = 299.45 mm`
  - flexible bar lengths `233 mm` and `215 mm`
  - `40 mm x 12 mm` cross section
  - `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`
- Added `kinematics/p2_fk.py` and FK tests.
- Added forward trimmed hemisphere target generation.
- Added bin-balanced target selection.
- Added repeat-balanced trial schedule generation.
- Added basic-target-hold-basic-hold minimum-jerk trajectory generation.
- Added matplotlib workspace, q-space, bin-count, and trajectory preview plots.
- Added STEP inspection scaffold for optional CAD tools.
- Added Vive calibration, settled dataset, and dataset validation scaffolds.
- Added verification report.
- Added documentation based on P1 lessons while treating P1/06 as incomplete reference, not ground truth.
