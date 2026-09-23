#pragma once

#include <Arduino.h>

// 3차원 벡터를 매우 가볍게 표현하기 위한 기본 자료형
struct Vector3f {
    float x;
    float y;
    float z;
};

// 월드-센서/월드-링크 회전을 표현하기 위한 사원수 자료형
struct Quaternionf {
    float w;
    float x;
    float y;
    float z;
};

// 각 IMU에서 한 주기마다 들어오는 입력 샘플
// q_world_from_sensor 는 기존 필터가 이미 계산한 자세 결과를 넣는 자리이다.
struct ImuPoseSample {
    Quaternionf q_world_from_sensor;
    Vector3f accel_mps2;
    Vector3f gyro_rad_s;
    bool orientation_valid;
};

enum ImuSensorId : uint8_t {
    kUpperImu1 = 0,
    kUpperImu2 = 1,
    kForearmImu1 = 2,
    kImuCount = 3,
};

// 센서 하나가 링크에 어떻게 붙어 있는지를 정의하는 고정 모델
struct ImuMountModel {
    // 링크 원점 기준 센서 중심 위치
    Vector3f sensor_position_in_link_m;

    // 센서 좌표계에서 링크 좌표계로 가는 고정 회전
    // 기준 자세 캘리브레이션 이후 q_link_from_sensor 에 저장된다.
    Quaternionf q_link_from_sensor;

    // 기준 자세에서 평균낸 자이로 바이어스
    Vector3f gyro_bias_rad_s;

    // 기준 자세 캘리브레이션이 끝났는지 표시
    bool reference_ready;
};

// 링크 하나에 대한 자세 추정 결과
struct LinkOrientationEstimate {
    Quaternionf q_world_from_link;
    Vector3f link_direction_world;
    Vector3f hinge_axis_world;
    float confidence;
};

// 최종적으로 외부에서 소비하는 관절/자세 추정 결과
struct JointAngleEstimate {
    bool valid;

    // 주 추정기: 힌지 평면 투영 기반
    float elbow_primary_deg;

    // 보조 추정기: 상대 사원수 기반
    float elbow_backup_deg;

    // 최종 권장 출력값
    float elbow_fused_deg;

    // 상완 링크의 중력 기준 기울기
    float upper_arm_tilt_deg;

    // 전완 링크의 중력 기준 기울기
    float forearm_tilt_deg;

    // 상완 IMU 2개의 자세 결과가 얼마나 일치하는지 보는 지표
    float upper_link_consistency_deg;

    // 전완 IMU가 링크 중심선 밖에 있기 때문에 발생하는 동적 가속도 영향 지표
    float forearm_dynamic_penalty_mps2;

    LinkOrientationEstimate upper_arm;
    LinkOrientationEstimate forearm;
};

// 기준 자세에서 여러 샘플의 사원수를 평균내기 위한 누산기
class QuaternionAccumulator {
public:
    QuaternionAccumulator();
    void reset();
    void push(const Quaternionf& q);
    Quaternionf mean() const;
    uint16_t count() const;

private:
    Quaternionf reference_;
    float sum_w_;
    float sum_x_;
    float sum_y_;
    float sum_z_;
    uint16_t count_;
};

class JointAngleEstimator {
public:
    JointAngleEstimator();

    // 내부 상태를 초기화하고 기본 모델로 되돌린다.
    void reset();

    // 기구학적으로 알려진 센서 위치를 기본값으로 설정한다.
    // 전완 IMU는 링크 상단에 붙어 있으므로 z 오프셋을 반드시 넣어야 한다.
    void configureDefaultMounts(float forearm_top_offset_m);

    // 기준 90도 자세에서 센서-링크 고정 회전과 자이로 바이어스를 잡기 위한 절차
    void beginReferenceCalibration();
    bool pushReferenceCalibrationSample(const ImuPoseSample samples[kImuCount]);
    bool finishReferenceCalibration();

    // 선택 절차: 실제 힌지축이 +Y 와 조금 어긋나 있을 때 상대 자이로로 힌지축을 재추정
    void beginElbowAxisCalibration();
    bool pushElbowAxisCalibrationSample(const ImuPoseSample samples[kImuCount], float dt_s);
    bool finishElbowAxisCalibration();

    // 실시간 추정 메인 함수
    bool update(const ImuPoseSample samples[kImuCount], float dt_s);

    const JointAngleEstimate& estimate() const;
    bool referenceReady() const;
    bool elbowAxisReady() const;

private:
    friend class QuaternionAccumulator;

    static constexpr float kGravityMps2 = 9.80665f;
    static constexpr float kRadToDeg = 57.2957795f;
    static constexpr float kDegToRad = 0.0174532925f;

    ImuMountModel mounts_[kImuCount];
    QuaternionAccumulator reference_quaternion_accumulators_[kImuCount];
    Vector3f reference_gyro_sum_[kImuCount];
    uint16_t reference_sample_count_;
    bool reference_mode_;

    // 상대 각속도 방향을 여러 번 모아 힌지축 공분산을 만든다.
    float elbow_axis_covariance_[3][3];
    uint16_t elbow_axis_sample_count_;
    bool elbow_axis_mode_;
    bool elbow_axis_ready_;

    // 힌지축을 상완 프레임과 전완 프레임에서 각각 저장한다.
    Vector3f elbow_axis_upper_frame_;
    Vector3f elbow_axis_forearm_frame_;

    // 기준 자세의 절대 팔꿈치 각도
    // 현재 문제 정의에서는 90도를 기준 영점으로 사용한다.
    float elbow_reference_angle_rad_;

    // 전완 IMU의 동적 가속도 보정량을 계산하기 위한 이전 자이로 저장값
    Vector3f previous_forearm_gyro_link_rad_s_;
    bool previous_forearm_gyro_valid_;

    JointAngleEstimate latest_estimate_;

    Quaternionf computeLinkQuaternion(const ImuPoseSample& sample, const ImuMountModel& mount) const;
    Vector3f computeLinkGyroRadS(const ImuPoseSample& sample, const ImuMountModel& mount) const;
    Quaternionf fuseUpperArmQuaternion(const Quaternionf& q_upper_1,
                                       const Quaternionf& q_upper_2,
                                       float accel_weight_upper_1,
                                       float accel_weight_upper_2,
                                       float* out_consistency_deg) const;
    float computeAccelConfidence(const ImuPoseSample& sample) const;
    float computeForearmDynamicPenalty(const ImuPoseSample& sample,
                                       const Vector3f& forearm_gyro_link_rad_s,
                                       float dt_s) const;
    Vector3f estimateRotationalSpecificForceMps2(const Vector3f& angular_velocity_link_rad_s,
                                                 const Vector3f& angular_acceleration_link_rad_s2,
                                                 const Vector3f& sensor_position_in_link_m) const;

    float estimateElbowAnglePrimaryRad(const Quaternionf& q_world_from_upper,
                                       const Quaternionf& q_world_from_forearm) const;
    float estimateElbowAngleBackupRad(const Quaternionf& q_world_from_upper,
                                      const Quaternionf& q_world_from_forearm) const;

    static Vector3f makeVector(float x, float y, float z);
    static Quaternionf makeQuaternion(float w, float x, float y, float z);
    static Vector3f add(const Vector3f& a, const Vector3f& b);
    static Vector3f subtract(const Vector3f& a, const Vector3f& b);
    static Vector3f scale(const Vector3f& value, float scale_value);
    static float dot(const Vector3f& a, const Vector3f& b);
    static Vector3f cross(const Vector3f& a, const Vector3f& b);
    static float norm(const Vector3f& value);
    static Vector3f normalize(const Vector3f& value);
    static Quaternionf quaternionNormalize(const Quaternionf& q);
    static Quaternionf quaternionConjugate(const Quaternionf& q);
    static Quaternionf quaternionMultiply(const Quaternionf& left, const Quaternionf& right);
    static float quaternionDot(const Quaternionf& a, const Quaternionf& b);
    static Vector3f rotateVector(const Quaternionf& q_world_from_local, const Vector3f& vector_local);
    static Quaternionf quaternionWeightedAverage(const Quaternionf& first,
                                                 const Quaternionf& second,
                                                 float first_weight,
                                                 float second_weight);
    static float quaternionAngleRad(const Quaternionf& a, const Quaternionf& b);
    static float signedAngleAroundAxisRad(const Vector3f& from_vector,
                                          const Vector3f& to_vector,
                                          const Vector3f& axis_world);
    static Vector3f projectToPlane(const Vector3f& value, const Vector3f& plane_normal);
    static float clampf(float value, float min_value, float max_value);
    static float wrapPi(float angle_rad);
    static Vector3f powerIterationMajorAxis(const float covariance[3][3]);
};
