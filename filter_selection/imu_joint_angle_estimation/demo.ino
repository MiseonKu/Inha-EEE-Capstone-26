#include <Arduino.h>

#include "Estimator.h"

namespace {

constexpr uint32_t kLoopPeriodMs = 10;
constexpr uint32_t kHumanPrintPeriodMs = 250;
constexpr uint32_t kViewerStreamPeriodMs = 50;
constexpr uint16_t kReferenceCalibrationTargetSamples = 300;
constexpr uint16_t kElbowAxisCalibrationTargetSamples = 220;
constexpr float kForearmTopOffsetMeters = 0.018f;
constexpr float kRadToDeg = 57.2957795f;

enum DemoMode : uint8_t {
    kModeIdle = 0,
    kModeReferenceCalibration = 1,
    kModeElbowAxisCalibration = 2,
    kModeRunContinuous = 3,
};

JointAngleEstimator g_estimator;
DemoMode g_mode = kModeIdle;
uint16_t g_mode_sample_count = 0;
uint32_t g_last_update_ms = 0;
uint32_t g_last_human_print_ms = 0;
uint32_t g_last_viewer_print_ms = 0;
bool g_viewer_stream_enabled = true;

// 이 함수는 기존 MPU-6050 필터 출력과 연결해야 한다.
// 각 센서에 대해 필터가 만든 사원수, 가속도, 자이로 값을 채우고
// 자세가 유효할 때 orientation_valid 를 true 로 바꾸면 된다.
bool readImuPoseSample(ImuSensorId sensor_id, ImuPoseSample* out_sample) {
    if (out_sample == nullptr) {
        return false;
    }

    *out_sample = {};
    out_sample->q_world_from_sensor = {1.0f, 0.0f, 0.0f, 0.0f};
    out_sample->accel_mps2 = {0.0f, 0.0f, 9.80665f};
    out_sample->gyro_rad_s = {0.0f, 0.0f, 0.0f};
    out_sample->orientation_valid = false;

    switch (sensor_id) {
        case kUpperImu1:
            // 여기에 상완 1번 IMU 필터 출력 연결
            break;
        case kUpperImu2:
            // 여기에 상완 2번 IMU 필터 출력 연결
            break;
        case kForearmImu1:
            // 여기에 전완 1번 IMU 필터 출력 연결
            break;
        default:
            return false;
    }

    return out_sample->orientation_valid;
}

// 3개 IMU 샘플을 한 번에 읽는다.
bool readAllImuSamples(ImuPoseSample samples[kImuCount]) {
    for (uint8_t index = 0; index < kImuCount; ++index) {
        if (!readImuPoseSample(static_cast<ImuSensorId>(index), &samples[index])) {
            return false;
        }
    }
    return true;
}

// 사람이 읽는 상태 메시지는 모두 # 으로 시작하게 만들어
// 호스트 뷰어가 데이터 줄과 쉽게 구분하도록 한다.
void printStatusLine(const __FlashStringHelper* message) {
    Serial.print(F("# "));
    Serial.println(message);
}

void printHelp() {
    Serial.println(F("# "));
    Serial.println(F("# === IMU Joint Estimator Demo ==="));
    Serial.println(F("# r : start reference calibration"));
    Serial.println(F("# e : start elbow-axis calibration"));
    Serial.println(F("# g : start continuous estimation"));
    Serial.println(F("# x : stop current mode"));
    Serial.println(F("# s : print estimate once"));
    Serial.println(F("# p : print viewer CSV header"));
    Serial.println(F("# v : toggle viewer CSV stream"));
    Serial.println(F("# h : help"));
    Serial.println(F("# "));
}

void printEstimate(const JointAngleEstimate& estimate) {
    if (!estimate.valid) {
        printStatusLine(F("estimate is not valid yet"));
        return;
    }

    Serial.print(F("# elbow_primary_deg="));
    Serial.print(estimate.elbow_primary_deg, 3);
    Serial.print(F(", elbow_backup_deg="));
    Serial.print(estimate.elbow_backup_deg, 3);
    Serial.print(F(", elbow_fused_deg="));
    Serial.print(estimate.elbow_fused_deg, 3);
    Serial.print(F(", upper_tilt_deg="));
    Serial.print(estimate.upper_arm_tilt_deg, 3);
    Serial.print(F(", forearm_tilt_deg="));
    Serial.print(estimate.forearm_tilt_deg, 3);
    Serial.print(F(", upper_consistency_deg="));
    Serial.print(estimate.upper_link_consistency_deg, 3);
    Serial.print(F(", forearm_dynamic_penalty="));
    Serial.println(estimate.forearm_dynamic_penalty_mps2, 3);
}

void printViewerHeader() {
    Serial.println(
        F("VH,millis,mode,estimate_valid,elbow_primary_deg,elbow_backup_deg,elbow_fused_deg,"
          "upper_tilt_deg,forearm_tilt_deg,upper_consistency_deg,forearm_dynamic_penalty_mps2,"
          "upper_confidence,forearm_confidence,"
          "upper_dir_x,upper_dir_y,upper_dir_z,"
          "forearm_dir_x,forearm_dir_y,forearm_dir_z,"
          "upper_hinge_x,upper_hinge_y,upper_hinge_z,"
          "forearm_hinge_x,forearm_hinge_y,forearm_hinge_z,"
          "u1_valid,u1_ax_mps2,u1_ay_mps2,u1_az_mps2,u1_gx_dps,u1_gy_dps,u1_gz_dps,"
          "u2_valid,u2_ax_mps2,u2_ay_mps2,u2_az_mps2,u2_gx_dps,u2_gy_dps,u2_gz_dps,"
          "f1_valid,f1_ax_mps2,f1_ay_mps2,f1_az_mps2,f1_gx_dps,f1_gy_dps,f1_gz_dps")
    );
}

void printCsvFloat(float value) {
    Serial.print(value, 6);
}

void printCsvVector3(const Vector3f& value) {
    printCsvFloat(value.x);
    Serial.print(',');
    printCsvFloat(value.y);
    Serial.print(',');
    printCsvFloat(value.z);
}

void printCsvSensorBlock(const ImuPoseSample& sample) {
    Serial.print(sample.orientation_valid ? 1 : 0);
    Serial.print(',');
    printCsvFloat(sample.accel_mps2.x);
    Serial.print(',');
    printCsvFloat(sample.accel_mps2.y);
    Serial.print(',');
    printCsvFloat(sample.accel_mps2.z);
    Serial.print(',');
    printCsvFloat(sample.gyro_rad_s.x * kRadToDeg);
    Serial.print(',');
    printCsvFloat(sample.gyro_rad_s.y * kRadToDeg);
    Serial.print(',');
    printCsvFloat(sample.gyro_rad_s.z * kRadToDeg);
}

// 실시간 뷰어가 읽는 데이터 줄
// 사람용 로그와 충돌하지 않도록 항상 VD 로 시작한다.
void printViewerFrame(const ImuPoseSample samples[kImuCount],
                      const JointAngleEstimate& estimate,
                      uint32_t now_ms) {
    Serial.print(F("VD,"));
    Serial.print(now_ms);
    Serial.print(',');
    Serial.print(static_cast<uint8_t>(g_mode));
    Serial.print(',');
    Serial.print(estimate.valid ? 1 : 0);
    Serial.print(',');

    printCsvFloat(estimate.elbow_primary_deg);
    Serial.print(',');
    printCsvFloat(estimate.elbow_backup_deg);
    Serial.print(',');
    printCsvFloat(estimate.elbow_fused_deg);
    Serial.print(',');
    printCsvFloat(estimate.upper_arm_tilt_deg);
    Serial.print(',');
    printCsvFloat(estimate.forearm_tilt_deg);
    Serial.print(',');
    printCsvFloat(estimate.upper_link_consistency_deg);
    Serial.print(',');
    printCsvFloat(estimate.forearm_dynamic_penalty_mps2);
    Serial.print(',');
    printCsvFloat(estimate.upper_arm.confidence);
    Serial.print(',');
    printCsvFloat(estimate.forearm.confidence);
    Serial.print(',');

    printCsvVector3(estimate.upper_arm.link_direction_world);
    Serial.print(',');
    printCsvVector3(estimate.forearm.link_direction_world);
    Serial.print(',');
    printCsvVector3(estimate.upper_arm.hinge_axis_world);
    Serial.print(',');
    printCsvVector3(estimate.forearm.hinge_axis_world);
    Serial.print(',');

    printCsvSensorBlock(samples[kUpperImu1]);
    Serial.print(',');
    printCsvSensorBlock(samples[kUpperImu2]);
    Serial.print(',');
    printCsvSensorBlock(samples[kForearmImu1]);
    Serial.println();
}

void startReferenceCalibration() {
    g_estimator.beginReferenceCalibration();
    g_mode = kModeReferenceCalibration;
    g_mode_sample_count = 0;
    printStatusLine(F("reference calibration started"));
}

void startElbowAxisCalibration() {
    if (!g_estimator.referenceReady()) {
        printStatusLine(F("reference calibration is required first"));
        return;
    }

    g_estimator.beginElbowAxisCalibration();
    g_mode = kModeElbowAxisCalibration;
    g_mode_sample_count = 0;
    printStatusLine(F("elbow-axis calibration started"));
}

void startContinuousRun() {
    if (!g_estimator.referenceReady()) {
        printStatusLine(F("reference calibration is required first"));
        return;
    }

    g_mode = kModeRunContinuous;
    g_last_viewer_print_ms = 0;
    printViewerHeader();
    printStatusLine(F("continuous estimation started"));
}

void stopMode() {
    g_mode = kModeIdle;
    g_mode_sample_count = 0;
    printStatusLine(F("current mode stopped"));
}

void toggleViewerStream() {
    g_viewer_stream_enabled = !g_viewer_stream_enabled;
    if (g_viewer_stream_enabled) {
        printViewerHeader();
        printStatusLine(F("viewer CSV stream enabled"));
    } else {
        printStatusLine(F("viewer CSV stream disabled"));
    }
}

void handleSerialCommand() {
    while (Serial.available() > 0) {
        const char command = static_cast<char>(Serial.read());

        if (command == 'r') {
            startReferenceCalibration();
        } else if (command == 'e') {
            startElbowAxisCalibration();
        } else if (command == 'g') {
            startContinuousRun();
        } else if (command == 'x') {
            stopMode();
        } else if (command == 's') {
            printEstimate(g_estimator.estimate());
        } else if (command == 'p') {
            printViewerHeader();
        } else if (command == 'v') {
            toggleViewerStream();
        } else if (command == 'h') {
            printHelp();
        }
    }
}

void processReferenceCalibration(const ImuPoseSample samples[kImuCount]) {
    if (g_estimator.pushReferenceCalibrationSample(samples)) {
        ++g_mode_sample_count;
    }

    if (g_mode_sample_count >= kReferenceCalibrationTargetSamples) {
        if (g_estimator.finishReferenceCalibration()) {
            printStatusLine(F("reference calibration completed"));
        } else {
            printStatusLine(F("reference calibration failed"));
        }
        g_mode = kModeIdle;
        g_mode_sample_count = 0;
    }
}

void processElbowAxisCalibration(const ImuPoseSample samples[kImuCount], float dt_s) {
    if (g_estimator.pushElbowAxisCalibrationSample(samples, dt_s)) {
        ++g_mode_sample_count;
    }

    if (g_mode_sample_count >= kElbowAxisCalibrationTargetSamples) {
        if (g_estimator.finishElbowAxisCalibration()) {
            printStatusLine(F("elbow-axis calibration completed"));
        } else {
            printStatusLine(F("elbow-axis calibration failed"));
        }
        g_mode = kModeIdle;
        g_mode_sample_count = 0;
    }
}

void maybePrintViewerFrame(const ImuPoseSample samples[kImuCount], uint32_t now_ms) {
    if (!g_viewer_stream_enabled) {
        return;
    }
    if (g_mode != kModeRunContinuous) {
        return;
    }
    if ((now_ms - g_last_viewer_print_ms) < kViewerStreamPeriodMs) {
        return;
    }

    const JointAngleEstimate& estimate = g_estimator.estimate();
    if (!estimate.valid) {
        return;
    }

    g_last_viewer_print_ms = now_ms;
    printViewerFrame(samples, estimate, now_ms);
}

void processRun(const ImuPoseSample samples[kImuCount], float dt_s, uint32_t now_ms) {
    if (!g_estimator.update(samples, dt_s)) {
        return;
    }

    if ((now_ms - g_last_human_print_ms) >= kHumanPrintPeriodMs) {
        g_last_human_print_ms = now_ms;
        printEstimate(g_estimator.estimate());
    }

    maybePrintViewerFrame(samples, now_ms);
}

}  // 익명 네임스페이스 종료

void setup() {
    Serial.begin(115200);
    delay(300);

    g_estimator.reset();
    g_estimator.configureDefaultMounts(kForearmTopOffsetMeters);
    g_last_update_ms = millis();
    g_last_human_print_ms = g_last_update_ms;
    g_last_viewer_print_ms = g_last_update_ms;

    printHelp();
    printViewerHeader();
}

void loop() {
    handleSerialCommand();

    const uint32_t now_ms = millis();
    if ((now_ms - g_last_update_ms) < kLoopPeriodMs) {
        return;
    }

    float dt_s = static_cast<float>(now_ms - g_last_update_ms) * 0.001f;
    if (dt_s <= 0.0f) {
        dt_s = 0.01f;
    }
    g_last_update_ms = now_ms;

    ImuPoseSample samples[kImuCount];
    if (!readAllImuSamples(samples)) {
        return;
    }

    switch (g_mode) {
        case kModeReferenceCalibration:
            processReferenceCalibration(samples);
            break;
        case kModeElbowAxisCalibration:
            processElbowAxisCalibration(samples, dt_s);
            break;
        case kModeRunContinuous:
            processRun(samples, dt_s, now_ms);
            break;
        case kModeIdle:
        default:
            // 대기 모드에서는 추정기 자체를 강제로 돌리지 않는다.
            // 필요하다면 여기에서 저속 상태 점검용 업데이트를 추가할 수 있다.
            break;
    }
}
