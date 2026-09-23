# 06 Simulation And Data Collect Preparation

AlphaF P1 한 링크 실험을 위한 ROS2/RViz2 기반 시뮬레이션 및 데이터 수집 준비 폴더입니다.

이 폴더의 목적은 실제 로봇을 바로 움직이는 것이 아니라, 실험 전에 다음 항목을 한 번에 준비하는 것입니다.

- P1 한 링크 모델의 rigid FK 기준 정리
- 목표점 target CSV 생성
- 반복 trial schedule CSV 생성
- minimum-jerk joint trajectory 생성
- ROS2 `robot_state_publisher` + RViz2 모델 표시
- target marker 표시
- Vive Tracker 좌표와 목표 좌표를 비교하기 위한 scaffold 제공

중요: 이 폴더에는 실제 모터 구동 코드가 들어있지 않습니다. 펌웨어 업로드, serial monitor, torque, INIT, calibration, MOVE, motor command는 수행하지 않습니다.

## 실험 기준

- stage: `P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL`
- 설치 상태: link 1만 장착, downstream joint/link 제거
- IMU: 이 phase에서는 사용하지 않음
- Vive Tracker: 실제 실험에서 link 1 끝단 ground-truth label 측정
- simulation: 변형을 시뮬레이션하지 않고 rigid reference target 좌표만 제공

Authoritative geometry:

- `L1_effective_m = 0.274123`
- flexible bar length: `0.233 m`
- adapter offset: `0.041123 m`
- material preset: `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`

기본 FK:

```text
rho = L1 * cos(q2)
z   = L1 * sin(q2)
x   = rho * cos(q1)
y   = rho * sin(q1)
```

기본 자세:

```text
q_basic = [q1, q2] = [0, pi/2]
```

## 폴더 구조

```text
06_simulation_and_data_collect_preparation/p1_one_link_experiment/
├── config/                 # P1 geometry/workspace/data collection config
├── kinematics/             # FK/IK 및 P1 rigid FK
├── sampling/               # target sampling, bin balancing, schedule logic
├── trajectory/             # minimum-jerk trajectory generation
├── scripts/                # CLI scripts and one-command helpers
├── tests/                  # Python unit tests
├── outputs/pre_imu_p1/     # 작은 target/schedule 산출물
└── simulation/
    ├── docs/               # frame/model/setup/verification 문서
    ├── generated/          # metadata/warnings
    ├── source_step/        # STEP 파일 위치 안내
    └── ros2_ws/            # ROS2 workspace
```

## 빠른 실행

Ubuntu + ROS2 + RViz2가 설치된 환경에서:

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment
bash scripts/p1_setup_local_deps.sh
```

의존성 설치 스크립트는 프로젝트 내부 `.python_deps/`에 Python 패키지를 설치합니다. 시스템 Python을 직접 변경하지 않기 위한 용도입니다.

target/schedule/trajectory 파일만 만들려면:

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment
bash scripts/p1_prepare_only.sh --target-count 120 --repeats 3
```

RViz2까지 실행하려면:

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment
source /opt/ros/$ROS_DISTRO/setup.bash
bash scripts/p1_run_demo.sh --target-count 80
```

위 명령은 다음을 수행합니다.

1. balanced target 생성
2. trial schedule 생성
3. `basic -> target -> hold -> basic -> hold` minimum-jerk trajectory 생성
4. ROS2 workspace build
5. RViz2에서 P1 one-link robot, target marker, trajectory 재생

## 포인트 개수 바꾸기

포인트 개수는 `--target-count`로 정합니다.

```bash
bash scripts/p1_prepare_only.sh --target-count 30 --repeats 3
bash scripts/p1_prepare_only.sh --target-count 80 --repeats 3
bash scripts/p1_prepare_only.sh --target-count 120 --repeats 3
```

target sampling은 Gaussian sampling이 아니라, 가능한 occupied yaw/pitch/height bin에 균등하게 배분되도록 설계되어 있습니다. 즉 특정 영역에 probability가 몰리지 않도록 constant probability에 가까운 balanced bin selection을 사용합니다.

## 개별 검증 명령

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment

python3 scripts/check_p1_fk.py

python3 scripts/generate_p1_forward_targets.py \
  --config config/p1_forward_workspace.yaml \
  --target-count 30 \
  --output outputs/pre_imu_p1/p1_targets_pilot.csv \
  --metadata-output outputs/pre_imu_p1/p1_targets_pilot_metadata.json

python3 scripts/generate_p1_trial_schedule.py \
  --targets outputs/pre_imu_p1/p1_targets_pilot.csv \
  --repeats 3 \
  --output outputs/pre_imu_p1/p1_trial_schedule.csv \
  --metadata-output outputs/pre_imu_p1/p1_trial_schedule_metadata.json

python3 scripts/generate_p1_trajectories.py \
  --schedule outputs/pre_imu_p1/p1_trial_schedule.csv \
  --output outputs/pre_imu_p1/p1_trajectories.csv

python3 -m pytest tests -q
```

ROS2 workspace:

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment/simulation/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
python3 -m pytest src/alphaf_p1_sim/test -q
ros2 launch alphaf_p1_sim state_publisher_p1.launch.py
```

다른 터미널에서 TF 확인:

```bash
cd 06_simulation_and_data_collect_preparation/p1_one_link_experiment/simulation/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 run tf2_ros tf2_echo base_link tool0
```

RViz2:

```bash
ros2 launch alphaf_p1_sim display_p1_one_link.launch.py
```

Gazebo/GZ:

```bash
ros2 launch alphaf_p1_sim gazebo_p1_one_link.launch.py
```

## 실제 Vive Tracker와 비교하는 흐름

simulation target은 바뀌지 않는 기준 좌표입니다. 실제 실험에서는 다음 값을 기록합니다.

- encoder에서 읽은 settled joint angle
- Vive Tracker에서 읽은 settled end-effector 좌표
- 현재 active target id / trial id

비교는 다음 순서로 합니다.

1. target CSV에서 목표 좌표 `p_target_base`를 읽습니다.
2. Vive world 좌표를 calibration matrix로 `base_link` 좌표로 변환합니다.
3. `error = p_vive_base - p_target_base`를 계산합니다.
4. settled window 구간의 평균/표준편차를 dataset으로 저장합니다.

관련 script:

- `scripts/calibrate_p1_vive_to_base.py`
- `scripts/build_p1_settled_dataset.py`
- `simulation/ros2_ws/src/alphaf_p1_sim/scripts/p1_vive_target_comparator.py`

## 실제 하드웨어 연동 시 반드시 수정/검증할 부분

이 폴더는 GitHub에 올려두기 위한 simulation-side/data-collection preparation 패키지입니다. ROS2/RViz2에서 target, rigid reference, trajectory, Vive 비교 scaffold까지 준비해두었지만, 실제 로봇과 Vive Tracker를 연결해 돌릴 때는 현장에서 반드시 확인하고 수정해야 하는 부분이 남아 있습니다.

### 구현되어 있는 것

- P1 one-link rigid FK 기준
- target point 생성
- balanced trial schedule 생성
- minimum-jerk trajectory 생성
- ROS2 one-link URDF/Xacro
- `robot_state_publisher` 기반 RViz2 표시
- target marker publisher
- trajectory player
- Vive 좌표와 target 좌표 비교용 scaffold
- Vive calibration 결과를 적용할 수 있는 비교 node
- settled dataset 생성을 위한 Python utility

### 실제 장비를 연결하면서 맞춰야 하는 것

1. 모터 제어 bridge

   이 폴더에는 실제 ST3215/SO-101 모터를 움직이는 command bridge를 넣지 않았습니다. 실제 구동은 기존 `03_SO-101_control` 또는 `05_SO-101_rtos_motor_driver` 쪽의 안전 gate, INIT, torque, calibration 정책과 연결해야 합니다. 이 패키지의 trajectory는 목표 joint angle sequence를 제공하는 역할이고, 실제 motor command 전송은 별도 bridge에서 담당해야 합니다.

2. encoder angle mapping

   실제 encoder에서 들어오는 값이 `q1`, `q2`와 정확히 어떻게 대응되는지 현장에서 확인해야 합니다.

   - encoder zero offset
   - motor direction/sign
   - radian 변환 scale
   - `q_basic = [0, pi/2]`와 실제 기본 자세 일치 여부
   - `q1=0`이 실제 전방인지
   - `q2=+pi/2`가 실제 수직 자세인지

   특히 motor sign convention은 실제 하드웨어 실행 전에 반드시 검증해야 합니다.

3. Vive Tracker ROS topic

   `p1_vive_target_comparator.py`는 기본적으로 `/vive_tracker/pose` 형태의 `geometry_msgs/PoseStamped` topic을 가정합니다. 실제 Vive driver, SteamVR/OpenVR bridge, tracker 이름에 따라 topic 이름과 frame 이름은 달라질 수 있습니다.

   실제 실행 전 확인할 것:

   - Vive pose topic 이름
   - pose message type
   - tracker frame id
   - tracking origin/world frame
   - Vive update rate
   - tracking loss 또는 outlier 처리 방식

4. Vive world to `base_link` calibration

   simulation target은 `base_link` 기준이고, Vive는 보통 Vive tracking world 기준입니다. 따라서 실제 비교 전에 Vive world 좌표를 `base_link`로 변환하는 calibration이 필요합니다.

   제공 script:

   ```bash
   python3 scripts/calibrate_p1_vive_to_base.py \
     --pairs calibration_pairs.csv \
     --output outputs/pre_imu_p1/p1_vive_to_base_calibration.json
   ```

   calibration pair CSV에는 같은 물리 점에 대한 `base_link` 좌표와 Vive world 좌표가 같이 들어가야 합니다. 현장에서는 calibration residual을 확인하고, residual이 크면 tracker mount, base frame 정의, scale/unit, 좌표축 방향을 다시 확인해야 합니다.

5. Tracker 장착 offset

   Vive Tracker 중심이 실제 `tool0/link1_tip`과 정확히 같지 않을 수 있습니다. tracker가 link 끝단에 어떤 브라켓으로 붙는지에 따라 offset이 생기므로, 필요하면 Vive pose에 tracker-to-tool0 offset 보정을 추가해야 합니다.

6. 시간 동기화와 settled window

   실제 dataset에서는 다음 timestamp가 서로 맞아야 합니다.

   - commanded target/trial id
   - encoder settled angle
   - Vive measured pose

   `build_p1_settled_dataset.py`는 settled window 평균을 만들 수 있는 scaffold지만, 실제 hold time, settle window 길이, outlier 제거 기준은 로봇을 돌리면서 조정해야 합니다.

7. trajectory duration과 안전 속도

   기본 trajectory는 minimum-jerk이고 target 도달 후 천천히 basic pose로 돌아오도록 되어 있습니다. 다만 실제 유연체 링크와 모터 응답에 따라 다음 값은 조정해야 합니다.

   - `move_duration_s`
   - `return_duration_s`
   - `target_hold_s`
   - `basic_hold_s`
   - sample rate
   - trial 사이 pause

   빠르게 움직이면 유연체 진동과 Vive label 오차가 커질 수 있으므로, 첫 실험은 아주 느린 속도로 시작해야 합니다.

8. trial abort/retry 정책

   실제 실험 중 tracking loss, motor fault, 과도한 error, 사람이 건드린 경우가 생길 수 있습니다. 이 repo에는 기본 schedule과 logging scaffold만 있으므로, 현장 운영에서는 trial skip/retry/invalid flag 정책을 추가해야 합니다.

9. ML dataset column 확정

   현재 phase에서는 IMU가 없으므로 기본 feature는 encoder angle, command angle, rigid FK, target 좌표, Vive measured 좌표입니다. 실제 모델 학습 전에 최종 CSV schema를 확정해야 합니다.

   권장 관계:

   ```text
   input  = encoder_q_settled, q_command, p_rigid_fk, p_target
   label  = p_vive_settled_base
   error  = p_vive_settled_base - p_target_base
   ```

   Vive Tracker는 controller feedback이 아니라 ground-truth label/evaluation signal로 사용해야 합니다.

### 현장 통합 전 체크리스트

- ROS2 package가 빌드되는가?
- RViz2에서 `base_link`, `link1_tip`, `tool0`가 보이는가?
- `tf2_echo base_link tool0`가 FK validation case와 맞는가?
- target marker가 `base_link` 기준으로 표시되는가?
- 실제 encoder 값이 q1/q2 radian으로 안정적으로 publish 되는가?
- Vive pose topic이 안정적으로 publish 되는가?
- Vive world to base calibration residual이 충분히 작은가?
- 실제 motor command bridge가 safety gate를 유지하는가?
- 첫 구동은 torque/속도/범위 제한을 걸고 매우 느리게 수행하는가?
- trial log에 invalid/retry/outlier flag를 남길 수 있는가?

## 포함하지 않은 파일

GitHub 커넥터 업로드를 가볍게 유지하기 위해 다음은 이 commit에 포함하지 않았습니다.

- ROS2 `build/`, `install/`, `log/`
- Python local dependency `.python_deps/`
- pytest/cache 파일
- 대형 workspace candidate CSV
- 대형 `outputs/pre_imu_p1/p1_trajectories.csv`
- RViz/Gazebo 스크린샷 및 screencast frame

`p1_trajectories.csv`는 위의 prepare 명령으로 즉시 재생성됩니다.

STEP 원본은 `simulation/source_step/SO101 Assembly-link11.step`에 포함했습니다. 이 파일은 kinematic authority가 아니라 visual geometry reference입니다. FK 길이와 joint 축은 config와 문서의 P1 authoritative geometry를 기준으로 합니다.

## 최근 로컬 검증 결과

로컬 Ubuntu + ROS2 lyrical 환경에서 확인한 결과:

- `python3 scripts/check_p1_fk.py`: PASS
- target generation: PASS
- schedule generation: PASS
- trajectory generation: PASS, smooth return-to-basic 기본 적용
- `python3 -m pytest tests -q`: 27 passed
- `colcon build --symlink-install`: PASS
- ROS2 package tests: 4 passed
- `robot_state_publisher`: 실행 확인
- `tf2_echo base_link tool0`: P1 검증 포즈 확인

자세한 기록은 `simulation/docs/verification_report.md`를 참고하세요.
