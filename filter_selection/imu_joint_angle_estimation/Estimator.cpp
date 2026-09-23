#include "Estimator.h"

#include <math.h>

// 기준 자세 캘리브레이션에서는 여러 사원수를 평균내야 한다.
// 사원수는 부호가 반대여도 같은 회전을 의미하므로,
// 기준 사원수와의 내적 부호를 맞춘 뒤 누적한다.
QuaternionAccumulator::QuaternionAccumulator()
    : reference_{1.0f, 0.0f, 0.0f, 0.0f},
      sum_w_(0.0f),
      sum_x_(0.0f),
      sum_y_(0.0f),
      sum_z_(0.0f),
      count_(0) {}

void QuaternionAccumulator::reset() {
    reference_ = {1.0f, 0.0f, 0.0f, 0.0f};
    sum_w_ = 0.0f;
    sum_x_ = 0.0f;
    sum_y_ = 0.0f;
    sum_z_ = 0.0f;
    count_ = 0;
}

void QuaternionAccumulator::push(const Quaternionf& q_input) {
    Quaternionf q = JointAngleEstimator::quaternionNormalize(q_input);
    if (count_ == 0) {
        reference_ = q;
    } else if (JointAngleEstimator::quaternionDot(reference_, q) < 0.0f) {
        q.w = -q.w;
        q.x = -q.x;
        q.y = -q.y;
        q.z = -q.z;
    }

    sum_w_ += q.w;
    sum_x_ += q.x;
    sum_y_ += q.y;
    sum_z_ += q.z;
    ++count_;
}

Quaternionf QuaternionAccumulator::mean() const {
    if (count_ == 0) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    return JointAngleEstimator::quaternionNormalize({sum_w_, sum_x_, sum_y_, sum_z_});
}

uint16_t QuaternionAccumulator::count() const {
    return count_;
}

JointAngleEstimator::JointAngleEstimator()
    : reference_sample_count_(0),
      reference_mode_(false),
      elbow_axis_sample_count_(0),
      elbow_axis_mode_(false),
      elbow_axis_ready_(false),
      elbow_axis_upper_frame_{0.0f, 1.0f, 0.0f},
      elbow_axis_forearm_frame_{0.0f, 1.0f, 0.0f},
      elbow_reference_angle_rad_(0.5f * PI),
      previous_forearm_gyro_link_rad_s_{0.0f, 0.0f, 0.0f},
      previous_forearm_gyro_valid_(false),
      latest_estimate_{} {
    reset();
}

// 추정기 전체 상태를 초기화한다.
// 이 함수는 기준 자세 캘리브레이션 결과도 모두 지우므로,
// 실제 운용 전에는 반드시 begin/finishReferenceCalibration 절차를 다시 수행해야 한다.
void JointAngleEstimator::reset() {
    for (uint8_t index = 0; index < kImuCount; ++index) {
        mounts_[index].sensor_position_in_link_m = makeVector(0.0f, 0.0f, 0.0f);
        mounts_[index].q_link_from_sensor = makeQuaternion(1.0f, 0.0f, 0.0f, 0.0f);
        mounts_[index].gyro_bias_rad_s = makeVector(0.0f, 0.0f, 0.0f);
        mounts_[index].reference_ready = false;
        reference_quaternion_accumulators_[index].reset();
        reference_gyro_sum_[index] = makeVector(0.0f, 0.0f, 0.0f);
    }

    reference_sample_count_ = 0;
    reference_mode_ = false;

    for (uint8_t row = 0; row < 3; ++row) {
        for (uint8_t col = 0; col < 3; ++col) {
            elbow_axis_covariance_[row][col] = 0.0f;
        }
    }
    elbow_axis_sample_count_ = 0;
    elbow_axis_mode_ = false;
    elbow_axis_ready_ = false;

    elbow_axis_upper_frame_ = makeVector(0.0f, 1.0f, 0.0f);
    elbow_axis_forearm_frame_ = makeVector(0.0f, 1.0f, 0.0f);
    elbow_reference_angle_rad_ = 0.5f * PI;

    previous_forearm_gyro_link_rad_s_ = makeVector(0.0f, 0.0f, 0.0f);
    previous_forearm_gyro_valid_ = false;

    latest_estimate_.valid = false;
    latest_estimate_.elbow_primary_deg = 0.0f;
    latest_estimate_.elbow_backup_deg = 0.0f;
    latest_estimate_.elbow_fused_deg = 0.0f;
    latest_estimate_.upper_arm_tilt_deg = 0.0f;
    latest_estimate_.forearm_tilt_deg = 0.0f;
    latest_estimate_.upper_link_consistency_deg = 0.0f;
    latest_estimate_.forearm_dynamic_penalty_mps2 = 0.0f;
    latest_estimate_.upper_arm.q_world_from_link = makeQuaternion(1.0f, 0.0f, 0.0f, 0.0f);
    latest_estimate_.upper_arm.link_direction_world = makeVector(0.0f, 0.0f, 1.0f);
    latest_estimate_.upper_arm.hinge_axis_world = makeVector(0.0f, 1.0f, 0.0f);
    latest_estimate_.upper_arm.confidence = 0.0f;
    latest_estimate_.forearm.q_world_from_link = makeQuaternion(1.0f, 0.0f, 0.0f, 0.0f);
    latest_estimate_.forearm.link_direction_world = makeVector(1.0f, 0.0f, 0.0f);
    latest_estimate_.forearm.hinge_axis_world = makeVector(0.0f, 1.0f, 0.0f);
    latest_estimate_.forearm.confidence = 0.0f;

    configureDefaultMounts(0.0f);
}

// 문제에서 주어진 센서 위치를 기본 탑재 모델로 넣는다.
// 상완 IMU 2개는 중심선 상에 있다고 보고 z 축 방향 거리만 사용한다.
// 전완 IMU는 링크 상단으로 올라가 있으므로 z 오프셋을 별도로 받는다.
void JointAngleEstimator::configureDefaultMounts(float forearm_top_offset_m) {
    mounts_[kUpperImu1].sensor_position_in_link_m = makeVector(0.0f, 0.0f, 0.070f);
    mounts_[kUpperImu2].sensor_position_in_link_m = makeVector(0.0f, 0.0f, 0.145f);
    mounts_[kForearmImu1].sensor_position_in_link_m = makeVector(0.145f, 0.0f, forearm_top_offset_m);
}

// 기준 90도 자세에서 각 센서의 자세 평균과 자이로 영점을 모으기 시작한다.
void JointAngleEstimator::beginReferenceCalibration() {
    reference_mode_ = true;
    reference_sample_count_ = 0;
    previous_forearm_gyro_valid_ = false;

    for (uint8_t index = 0; index < kImuCount; ++index) {
        reference_quaternion_accumulators_[index].reset();
        reference_gyro_sum_[index] = makeVector(0.0f, 0.0f, 0.0f);
    }
}

// 모든 센서가 기준 자세에서 안정적으로 고정되어 있다고 가정하고 샘플을 누적한다.
// orientation_valid 가 하나라도 false 이면 그 샘플은 버린다.
bool JointAngleEstimator::pushReferenceCalibrationSample(const ImuPoseSample samples[kImuCount]) {
    if (!reference_mode_) {
        return false;
    }

    for (uint8_t index = 0; index < kImuCount; ++index) {
        if (!samples[index].orientation_valid) {
            return false;
        }
    }

    for (uint8_t index = 0; index < kImuCount; ++index) {
        reference_quaternion_accumulators_[index].push(samples[index].q_world_from_sensor);
        reference_gyro_sum_[index] = add(reference_gyro_sum_[index], samples[index].gyro_rad_s);
    }

    ++reference_sample_count_;
    return true;
}

// 누적된 평균 사원수를 센서-링크 고정 회전으로 저장한다.
// 현재 좌표계 정의에서는 기준 자세에서 링크 프레임이 월드 프레임과 일치하도록 두었기 때문에,
// 기준 자세의 월드-센서 사원수 평균 자체가 곧 q_link_from_sensor 이 된다.
bool JointAngleEstimator::finishReferenceCalibration() {
    if (!reference_mode_ || reference_sample_count_ == 0) {
        return false;
    }

    for (uint8_t index = 0; index < kImuCount; ++index) {
        mounts_[index].q_link_from_sensor = reference_quaternion_accumulators_[index].mean();
        mounts_[index].gyro_bias_rad_s = scale(reference_gyro_sum_[index], 1.0f / reference_sample_count_);
        mounts_[index].reference_ready = true;
    }

    reference_mode_ = false;
    elbow_reference_angle_rad_ = 0.5f * PI;
    return true;
}

// 실제 조립에서 힌지축이 +Y 와 어긋날 수 있으므로 선택적으로 축 재보정을 시작한다.
void JointAngleEstimator::beginElbowAxisCalibration() {
    elbow_axis_mode_ = true;
    elbow_axis_ready_ = false;
    elbow_axis_sample_count_ = 0;

    for (uint8_t row = 0; row < 3; ++row) {
        for (uint8_t col = 0; col < 3; ++col) {
            elbow_axis_covariance_[row][col] = 0.0f;
        }
    }
}

// 상완과 전완의 상대 각속도 방향을 여러 번 관측해서 힌지축 후보를 모은다.
// 순수 굽힘 동작을 몇 번 반복하면 상대 각속도 방향이 힌지축 쪽으로 모이게 된다.
bool JointAngleEstimator::pushElbowAxisCalibrationSample(const ImuPoseSample samples[kImuCount], float dt_s) {
    if (!elbow_axis_mode_ || !referenceReady() || dt_s <= 0.0f) {
        return false;
    }

    Quaternionf q_world_from_upper_1 = computeLinkQuaternion(samples[kUpperImu1], mounts_[kUpperImu1]);
    Quaternionf q_world_from_upper_2 = computeLinkQuaternion(samples[kUpperImu2], mounts_[kUpperImu2]);
    Quaternionf q_world_from_forearm = computeLinkQuaternion(samples[kForearmImu1], mounts_[kForearmImu1]);

    float upper_consistency_deg = 0.0f;
    Quaternionf q_world_from_upper = fuseUpperArmQuaternion(
        q_world_from_upper_1,
        q_world_from_upper_2,
        1.0f,
        1.0f,
        &upper_consistency_deg
    );

    Vector3f gyro_upper_1_link = computeLinkGyroRadS(samples[kUpperImu1], mounts_[kUpperImu1]);
    Vector3f gyro_upper_2_link = computeLinkGyroRadS(samples[kUpperImu2], mounts_[kUpperImu2]);
    Vector3f gyro_forearm_link = computeLinkGyroRadS(samples[kForearmImu1], mounts_[kForearmImu1]);

    // 상완 링크는 동일 강체이므로 두 IMU의 각속도를 평균해 링크 각속도로 본다.
    Vector3f gyro_upper_link = scale(add(gyro_upper_1_link, gyro_upper_2_link), 0.5f);
    Quaternionf q_upper_from_forearm = quaternionMultiply(quaternionConjugate(q_world_from_upper), q_world_from_forearm);
    Vector3f gyro_forearm_in_upper = rotateVector(q_upper_from_forearm, gyro_forearm_link);
    Vector3f relative_gyro_upper = subtract(gyro_forearm_in_upper, gyro_upper_link);

    float relative_gyro_norm = norm(relative_gyro_upper);
    if (relative_gyro_norm < 0.15f) {
        return false;
    }

    Vector3f axis_candidate = normalize(relative_gyro_upper);

    elbow_axis_covariance_[0][0] += axis_candidate.x * axis_candidate.x;
    elbow_axis_covariance_[0][1] += axis_candidate.x * axis_candidate.y;
    elbow_axis_covariance_[0][2] += axis_candidate.x * axis_candidate.z;
    elbow_axis_covariance_[1][0] += axis_candidate.y * axis_candidate.x;
    elbow_axis_covariance_[1][1] += axis_candidate.y * axis_candidate.y;
    elbow_axis_covariance_[1][2] += axis_candidate.y * axis_candidate.z;
    elbow_axis_covariance_[2][0] += axis_candidate.z * axis_candidate.x;
    elbow_axis_covariance_[2][1] += axis_candidate.z * axis_candidate.y;
    elbow_axis_covariance_[2][2] += axis_candidate.z * axis_candidate.z;

    ++elbow_axis_sample_count_;
    return true;
}

// 상대 각속도 방향들의 공분산에서 주축을 뽑아 힌지축으로 사용한다.
bool JointAngleEstimator::finishElbowAxisCalibration() {
    if (!elbow_axis_mode_ || elbow_axis_sample_count_ == 0) {
        return false;
    }

    Vector3f major_axis = powerIterationMajorAxis(elbow_axis_covariance_);
    if (dot(major_axis, makeVector(0.0f, 1.0f, 0.0f)) < 0.0f) {
        major_axis = scale(major_axis, -1.0f);
    }

    elbow_axis_upper_frame_ = major_axis;
    elbow_axis_forearm_frame_ = major_axis;
    elbow_axis_mode_ = false;
    elbow_axis_ready_ = true;
    elbow_reference_angle_rad_ = 0.5f * PI;
    return true;
}

// 실시간 추정 메인 루프
// 1) 센서 자세를 링크 자세로 변환
// 2) 상완 2개 IMU를 일관성 기반으로 융합
// 3) 전완 동적 가속도 패널티 계산
// 4) 팔꿈치 각도를 주/보조 추정기로 계산
// 5) 필요한 진단 지표와 함께 결과를 저장
bool JointAngleEstimator::update(const ImuPoseSample samples[kImuCount], float dt_s) {
    if (!referenceReady()) {
        latest_estimate_.valid = false;
        return false;
    }

    for (uint8_t index = 0; index < kImuCount; ++index) {
        if (!samples[index].orientation_valid) {
            latest_estimate_.valid = false;
            return false;
        }
    }

    Quaternionf q_world_from_upper_1 = computeLinkQuaternion(samples[kUpperImu1], mounts_[kUpperImu1]);
    Quaternionf q_world_from_upper_2 = computeLinkQuaternion(samples[kUpperImu2], mounts_[kUpperImu2]);
    Quaternionf q_world_from_forearm = computeLinkQuaternion(samples[kForearmImu1], mounts_[kForearmImu1]);

    float accel_conf_upper_1 = computeAccelConfidence(samples[kUpperImu1]);
    float accel_conf_upper_2 = computeAccelConfidence(samples[kUpperImu2]);
    float upper_consistency_deg = 0.0f;

    Quaternionf q_world_from_upper = fuseUpperArmQuaternion(
        q_world_from_upper_1,
        q_world_from_upper_2,
        accel_conf_upper_1,
        accel_conf_upper_2,
        &upper_consistency_deg
    );

    // 좌표계 정의:
    // 상완은 +Z 가 링크 진행 방향, 전완은 +X 가 링크 진행 방향이다.
    Vector3f upper_link_direction_world = normalize(rotateVector(q_world_from_upper, makeVector(0.0f, 0.0f, 1.0f)));
    Vector3f forearm_link_direction_world = normalize(rotateVector(q_world_from_forearm, makeVector(1.0f, 0.0f, 0.0f)));
    Vector3f upper_hinge_axis_world = normalize(rotateVector(q_world_from_upper, elbow_axis_upper_frame_));
    Vector3f forearm_hinge_axis_world = normalize(rotateVector(q_world_from_forearm, elbow_axis_forearm_frame_));

    Vector3f gyro_forearm_link = computeLinkGyroRadS(samples[kForearmImu1], mounts_[kForearmImu1]);
    float forearm_dynamic_penalty = computeForearmDynamicPenalty(samples[kForearmImu1], gyro_forearm_link, dt_s);

    float forearm_confidence = clampf(
        computeAccelConfidence(samples[kForearmImu1]) / (1.0f + 0.2f * forearm_dynamic_penalty),
        0.0f,
        1.0f
    );

    float elbow_primary_rad = estimateElbowAnglePrimaryRad(q_world_from_upper, q_world_from_forearm);
    float elbow_backup_rad = estimateElbowAngleBackupRad(q_world_from_upper, q_world_from_forearm);

    // 주 추정기와 보조 추정기가 크게 다르면 주 추정기를 그대로 사용한다.
    // 두 값이 충분히 가깝다면 보조 추정기를 조금 섞어서 급격한 튐을 줄인다.
    float backup_gap_rad = fabsf(wrapPi(elbow_primary_rad - elbow_backup_rad));
    float elbow_fused_rad = elbow_primary_rad;
    if (backup_gap_rad < (12.0f * kDegToRad)) {
        elbow_fused_rad = 0.85f * elbow_primary_rad + 0.15f * elbow_backup_rad;
    }

    float upper_arm_tilt_deg = acosf(clampf(dot(upper_link_direction_world, makeVector(0.0f, 0.0f, 1.0f)), -1.0f, 1.0f)) * kRadToDeg;
    float forearm_tilt_deg = asinf(clampf(dot(forearm_link_direction_world, makeVector(0.0f, 0.0f, 1.0f)), -1.0f, 1.0f)) * kRadToDeg;

    latest_estimate_.valid = true;
    latest_estimate_.elbow_primary_deg = elbow_primary_rad * kRadToDeg;
    latest_estimate_.elbow_backup_deg = elbow_backup_rad * kRadToDeg;
    latest_estimate_.elbow_fused_deg = elbow_fused_rad * kRadToDeg;
    latest_estimate_.upper_arm_tilt_deg = upper_arm_tilt_deg;
    latest_estimate_.forearm_tilt_deg = forearm_tilt_deg;
    latest_estimate_.upper_link_consistency_deg = upper_consistency_deg;
    latest_estimate_.forearm_dynamic_penalty_mps2 = forearm_dynamic_penalty;

    latest_estimate_.upper_arm.q_world_from_link = q_world_from_upper;
    latest_estimate_.upper_arm.link_direction_world = upper_link_direction_world;
    latest_estimate_.upper_arm.hinge_axis_world = upper_hinge_axis_world;
    latest_estimate_.upper_arm.confidence = 0.5f * (accel_conf_upper_1 + accel_conf_upper_2);

    latest_estimate_.forearm.q_world_from_link = q_world_from_forearm;
    latest_estimate_.forearm.link_direction_world = forearm_link_direction_world;
    latest_estimate_.forearm.hinge_axis_world = forearm_hinge_axis_world;
    latest_estimate_.forearm.confidence = forearm_confidence;

    return true;
}

const JointAngleEstimate& JointAngleEstimator::estimate() const {
    return latest_estimate_;
}

bool JointAngleEstimator::referenceReady() const {
    for (uint8_t index = 0; index < kImuCount; ++index) {
        if (!mounts_[index].reference_ready) {
            return false;
        }
    }
    return true;
}

bool JointAngleEstimator::elbowAxisReady() const {
    return elbow_axis_ready_;
}

Quaternionf JointAngleEstimator::computeLinkQuaternion(const ImuPoseSample& sample, const ImuMountModel& mount) const {
    // q_world_from_sensor 와 기준 자세에서 얻은 q_link_from_sensor 를 이용해
    // 현재의 월드-링크 회전을 만든다.
    Quaternionf q_world_from_sensor = quaternionNormalize(sample.q_world_from_sensor);
    Quaternionf q_sensor_from_link = quaternionConjugate(mount.q_link_from_sensor);
    return quaternionNormalize(quaternionMultiply(q_world_from_sensor, q_sensor_from_link));
}

Vector3f JointAngleEstimator::computeLinkGyroRadS(const ImuPoseSample& sample, const ImuMountModel& mount) const {
    // 자이로는 벡터량이므로 센서 프레임에서 링크 프레임으로 회전시켜 사용한다.
    Quaternionf q_link_from_sensor = mount.q_link_from_sensor;
    Vector3f gyro_without_bias = subtract(sample.gyro_rad_s, mount.gyro_bias_rad_s);
    return rotateVector(q_link_from_sensor, gyro_without_bias);
}

Quaternionf JointAngleEstimator::fuseUpperArmQuaternion(const Quaternionf& q_upper_1,
                                                        const Quaternionf& q_upper_2,
                                                        float accel_weight_upper_1,
                                                        float accel_weight_upper_2,
                                                        float* out_consistency_deg) const {
    // 상완 2개 IMU가 서로 다르게 보이면 일단 그 자체를 진단 지표로 남긴다.
    // 동시에 가속도 신뢰도와 두 IMU의 일치도를 함께 반영해 평균 자세를 만든다.
    float consistency_rad = quaternionAngleRad(q_upper_1, q_upper_2);
    if (out_consistency_deg != nullptr) {
        *out_consistency_deg = consistency_rad * kRadToDeg;
    }

    float consistency_weight = 1.0f / (1.0f + consistency_rad * 3.0f);
    float weight_1 = clampf(accel_weight_upper_1 * consistency_weight, 0.05f, 1.0f);
    float weight_2 = clampf(accel_weight_upper_2 * consistency_weight, 0.05f, 1.0f);
    return quaternionWeightedAverage(q_upper_1, q_upper_2, weight_1, weight_2);
}

float JointAngleEstimator::computeAccelConfidence(const ImuPoseSample& sample) const {
    // 가속도 크기가 중력 크기에서 많이 벗어날수록
    // 해당 순간의 기울기 정보는 동적 가속도에 오염되었을 가능성이 크다.
    float accel_norm = norm(sample.accel_mps2);
    float accel_error = fabsf(accel_norm - kGravityMps2);
    return clampf(1.0f / (1.0f + 0.35f * accel_error), 0.05f, 1.0f);
}

float JointAngleEstimator::computeForearmDynamicPenalty(const ImuPoseSample& sample,
                                                        const Vector3f& forearm_gyro_link_rad_s,
                                                        float dt_s) const {
    // 전완 IMU는 링크 중심선이 아니라 상단에 붙어 있으므로
    // 회전만 있어도 접선/구심 가속도가 생긴다.
    // 이를 추정해서 전완 accelerometer 신뢰도를 낮추는 패널티로 사용한다.
    if (dt_s <= 0.0f || !previous_forearm_gyro_valid_) {
        previous_forearm_gyro_link_rad_s_ = forearm_gyro_link_rad_s;
        previous_forearm_gyro_valid_ = true;
        return fabsf(norm(sample.accel_mps2) - kGravityMps2);
    }

    Vector3f angular_acceleration_link_rad_s2 = scale(
        subtract(forearm_gyro_link_rad_s, previous_forearm_gyro_link_rad_s_),
        1.0f / dt_s
    );
    previous_forearm_gyro_link_rad_s_ = forearm_gyro_link_rad_s;

    Vector3f rotational_specific_force = estimateRotationalSpecificForceMps2(
        forearm_gyro_link_rad_s,
        angular_acceleration_link_rad_s2,
        mounts_[kForearmImu1].sensor_position_in_link_m
    );

    float accel_norm_error = fabsf(norm(sample.accel_mps2) - kGravityMps2);
    return accel_norm_error + norm(rotational_specific_force);
}

Vector3f JointAngleEstimator::estimateRotationalSpecificForceMps2(const Vector3f& angular_velocity_link_rad_s,
                                                                  const Vector3f& angular_acceleration_link_rad_s2,
                                                                  const Vector3f& sensor_position_in_link_m) const {
    // a = alpha x r + omega x (omega x r)
    // 링크 원점이 아닌 위치에 붙은 센서가 회전 때문에 추가로 받는 가속도 성분
    Vector3f tangential = cross(angular_acceleration_link_rad_s2, sensor_position_in_link_m);
    Vector3f centripetal = cross(angular_velocity_link_rad_s, cross(angular_velocity_link_rad_s, sensor_position_in_link_m));
    return add(tangential, centripetal);
}

float JointAngleEstimator::estimateElbowAnglePrimaryRad(const Quaternionf& q_world_from_upper,
                                                        const Quaternionf& q_world_from_forearm) const {
    // 주 추정기:
    // 1) 상완/전완 링크 방향을 힌지축에 수직인 평면으로 투영
    // 2) 그 평면에서의 부호 있는 각도를 계산
    // 3) 기준 자세의 90도를 빼서 영점 기준 팔꿈치 각도로 만든다.
    Vector3f axis_world = normalize(rotateVector(q_world_from_upper, elbow_axis_upper_frame_));
    Vector3f upper_direction_world = normalize(rotateVector(q_world_from_upper, makeVector(0.0f, 0.0f, 1.0f)));
    Vector3f forearm_direction_world = normalize(rotateVector(q_world_from_forearm, makeVector(1.0f, 0.0f, 0.0f)));

    Vector3f upper_projected = normalize(projectToPlane(upper_direction_world, axis_world));
    Vector3f forearm_projected = normalize(projectToPlane(forearm_direction_world, axis_world));
    if (norm(upper_projected) < 1.0e-5f || norm(forearm_projected) < 1.0e-5f) {
        return 0.0f;
    }

    float absolute_angle = signedAngleAroundAxisRad(upper_projected, forearm_projected, axis_world);
    return wrapPi(absolute_angle - elbow_reference_angle_rad_);
}

float JointAngleEstimator::estimateElbowAngleBackupRad(const Quaternionf& q_world_from_upper,
                                                       const Quaternionf& q_world_from_forearm) const {
    // 보조 추정기:
    // 상대 사원수를 상완 프레임에서 계산한 뒤,
    // 힌지축 성분만 남기는 방식으로 팔꿈치 회전을 근사한다.
    Quaternionf q_upper_from_forearm = quaternionMultiply(quaternionConjugate(q_world_from_upper), q_world_from_forearm);
    Quaternionf q_relative = quaternionNormalize(q_upper_from_forearm);

    Vector3f axis_upper = normalize(elbow_axis_upper_frame_);
    Vector3f vector_part = makeVector(q_relative.x, q_relative.y, q_relative.z);
    Vector3f vector_twist = scale(axis_upper, dot(vector_part, axis_upper));

    Quaternionf q_twist = quaternionNormalize(makeQuaternion(q_relative.w, vector_twist.x, vector_twist.y, vector_twist.z));
    float vector_norm = sqrtf((q_twist.x * q_twist.x) + (q_twist.y * q_twist.y) + (q_twist.z * q_twist.z));
    float twist_angle = 2.0f * atan2f(vector_norm, q_twist.w);

    float sign_value = dot(makeVector(q_twist.x, q_twist.y, q_twist.z), axis_upper);
    if (sign_value < 0.0f) {
        twist_angle = -twist_angle;
    }

    return wrapPi(twist_angle - elbow_reference_angle_rad_);
}

Vector3f JointAngleEstimator::makeVector(float x, float y, float z) {
    Vector3f out{x, y, z};
    return out;
}

Quaternionf JointAngleEstimator::makeQuaternion(float w, float x, float y, float z) {
    Quaternionf out{w, x, y, z};
    return out;
}

Vector3f JointAngleEstimator::add(const Vector3f& a, const Vector3f& b) {
    return makeVector(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vector3f JointAngleEstimator::subtract(const Vector3f& a, const Vector3f& b) {
    return makeVector(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vector3f JointAngleEstimator::scale(const Vector3f& value, float scale_value) {
    return makeVector(value.x * scale_value, value.y * scale_value, value.z * scale_value);
}

float JointAngleEstimator::dot(const Vector3f& a, const Vector3f& b) {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

Vector3f JointAngleEstimator::cross(const Vector3f& a, const Vector3f& b) {
    return makeVector(
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z),
        (a.x * b.y) - (a.y * b.x)
    );
}

float JointAngleEstimator::norm(const Vector3f& value) {
    return sqrtf(dot(value, value));
}

Vector3f JointAngleEstimator::normalize(const Vector3f& value) {
    float value_norm = norm(value);
    if (value_norm < 1.0e-8f) {
        return makeVector(0.0f, 0.0f, 0.0f);
    }
    return scale(value, 1.0f / value_norm);
}

Quaternionf JointAngleEstimator::quaternionNormalize(const Quaternionf& q) {
    float value_norm = sqrtf((q.w * q.w) + (q.x * q.x) + (q.y * q.y) + (q.z * q.z));
    if (value_norm < 1.0e-8f) {
        return makeQuaternion(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return makeQuaternion(q.w / value_norm, q.x / value_norm, q.y / value_norm, q.z / value_norm);
}

Quaternionf JointAngleEstimator::quaternionConjugate(const Quaternionf& q) {
    return makeQuaternion(q.w, -q.x, -q.y, -q.z);
}

Quaternionf JointAngleEstimator::quaternionMultiply(const Quaternionf& left, const Quaternionf& right) {
    return makeQuaternion(
        (left.w * right.w) - (left.x * right.x) - (left.y * right.y) - (left.z * right.z),
        (left.w * right.x) + (left.x * right.w) + (left.y * right.z) - (left.z * right.y),
        (left.w * right.y) - (left.x * right.z) + (left.y * right.w) + (left.z * right.x),
        (left.w * right.z) + (left.x * right.y) - (left.y * right.x) + (left.z * right.w)
    );
}

float JointAngleEstimator::quaternionDot(const Quaternionf& a, const Quaternionf& b) {
    return (a.w * b.w) + (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

Vector3f JointAngleEstimator::rotateVector(const Quaternionf& q_world_from_local, const Vector3f& vector_local) {
    Quaternionf vector_quaternion = makeQuaternion(0.0f, vector_local.x, vector_local.y, vector_local.z);
    Quaternionf rotated = quaternionMultiply(
        quaternionMultiply(q_world_from_local, vector_quaternion),
        quaternionConjugate(q_world_from_local)
    );
    return makeVector(rotated.x, rotated.y, rotated.z);
}

Quaternionf JointAngleEstimator::quaternionWeightedAverage(const Quaternionf& first,
                                                           const Quaternionf& second,
                                                           float first_weight,
                                                           float second_weight) {
    // 사원수 평균 전에는 두 사원수의 부호 반전을 맞춰야 불연속 점프를 피할 수 있다.
    Quaternionf aligned_second = second;
    if (quaternionDot(first, second) < 0.0f) {
        aligned_second.w = -aligned_second.w;
        aligned_second.x = -aligned_second.x;
        aligned_second.y = -aligned_second.y;
        aligned_second.z = -aligned_second.z;
    }

    Quaternionf mixed = makeQuaternion(
        (first_weight * first.w) + (second_weight * aligned_second.w),
        (first_weight * first.x) + (second_weight * aligned_second.x),
        (first_weight * first.y) + (second_weight * aligned_second.y),
        (first_weight * first.z) + (second_weight * aligned_second.z)
    );
    return quaternionNormalize(mixed);
}

float JointAngleEstimator::quaternionAngleRad(const Quaternionf& a, const Quaternionf& b) {
    // 사원수 간 최단 회전 차이
    float cosine_half_angle = fabsf(quaternionDot(quaternionNormalize(a), quaternionNormalize(b)));
    cosine_half_angle = clampf(cosine_half_angle, -1.0f, 1.0f);
    return 2.0f * acosf(cosine_half_angle);
}

float JointAngleEstimator::signedAngleAroundAxisRad(const Vector3f& from_vector,
                                                    const Vector3f& to_vector,
                                                    const Vector3f& axis_world) {
    Vector3f axis_normalized = normalize(axis_world);
    float sine_term = dot(axis_normalized, cross(from_vector, to_vector));
    float cosine_term = dot(from_vector, to_vector);
    return atan2f(sine_term, cosine_term);
}

Vector3f JointAngleEstimator::projectToPlane(const Vector3f& value, const Vector3f& plane_normal) {
    // 힌지축 방향 성분을 제거해서 굽힘 평면에만 남긴다.
    Vector3f normal = normalize(plane_normal);
    return subtract(value, scale(normal, dot(value, normal)));
}

float JointAngleEstimator::clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

float JointAngleEstimator::wrapPi(float angle_rad) {
    while (angle_rad > PI) {
        angle_rad -= 2.0f * PI;
    }
    while (angle_rad < -PI) {
        angle_rad += 2.0f * PI;
    }
    return angle_rad;
}

Vector3f JointAngleEstimator::powerIterationMajorAxis(const float covariance[3][3]) {
    // 간단한 power iteration 으로 공분산 행렬의 최대 고유벡터를 찾는다.
    // Arduino 환경을 고려해 무거운 행렬 라이브러리 없이 구현했다.
    Vector3f vector = makeVector(0.0f, 1.0f, 0.0f);

    for (uint8_t iteration = 0; iteration < 12; ++iteration) {
        Vector3f multiplied = makeVector(
            (covariance[0][0] * vector.x) + (covariance[0][1] * vector.y) + (covariance[0][2] * vector.z),
            (covariance[1][0] * vector.x) + (covariance[1][1] * vector.y) + (covariance[1][2] * vector.z),
            (covariance[2][0] * vector.x) + (covariance[2][1] * vector.y) + (covariance[2][2] * vector.z)
        );
        vector = normalize(multiplied);
    }

    if (norm(vector) < 1.0e-6f) {
        return makeVector(0.0f, 1.0f, 0.0f);
    }
    return vector;
}
