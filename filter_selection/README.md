# IMU Filter Selection

This folder contains the MPU-6050 acquisition tools, the recorded straight-pose
filter comparison data, and the Unit 1 IMU joint-angle estimator.

## Folder Map

| Folder | Role |
|---|---|
| `mpu6050_data_pipeline/` | Arduino sketches and Python tools for acquisition, calibration, plotting, pose filtering, and stability analysis. |
| `filter_results/` | Recorded trials, summary metrics, figures, and the result-description document. |
| `imu_joint_angle_estimation/` | Unit 1 three-IMU link-orientation and elbow-angle estimator, demo sketch, and live viewer. |

`imu_joint_angle_estimation/` is explicitly for Unit 1. This folder does not
contain a Unit 2 IMU implementation.

## Filter Comparison Data

The analysis uses the five 20-recording directories under
`filter_results/experiment_data/`, for 100 source CSV recordings in total.
Every source CSV contains simultaneous output columns for all five filters:
unfiltered, complementary, Kalman, Mahony, and Madgwick. Consequently, the
summary reports 100 evaluated trials for each filter, not 20.

`total_straight/` contains aggregate presentation material and is not consumed
by the stability-analysis script.

## Rebuild The Filter Summary

From the package root:

```powershell
python filter_selection\mpu6050_data_pipeline\plot_filter_stability.py
```

The default input is `filter_selection/filter_results/experiment_data/`, and
the generated summary and figures are written to
`filter_selection/filter_results/`. Lower stability scores indicate less
jitter and endpoint drift under the script's composite metric.

## Dependencies

Install the package dependencies from `environment/requirements.txt`.
The live acquisition tools additionally use `pyserial`, which is included in
that file. Arduino sketches require the board and MPU-6050 libraries configured
for the target hardware.

The scripts that open a serial port or send calibration commands operate on
real hardware. Review the selected port and command before running them.
