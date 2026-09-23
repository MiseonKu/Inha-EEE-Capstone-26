# Data Collection P2

Future real logs should preserve:

- `target_id`
- `trial_id`
- q command values
- encoder q values if available
- rigid target base coordinates
- Vive Tracker raw samples
- Vive settled labels
- geometry metadata

Vive-to-base calibration should be saved as JSON and treated separately from model input. The settled dataset should include quality flags for tracker availability, alignment gaps, and minimum samples per target.

Vive Tracker is label/evaluation only. It is not control feedback.
