from __future__ import annotations

import math
from dataclasses import dataclass


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

EPSILON = 1e-9
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
IDENTITY_QUATERNION: Quaternion = (1.0, 0.0, 0.0, 0.0)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def wrap_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def vector_norm(vector: Vector3) -> float:
    x, y, z = vector
    return math.sqrt((x * x) + (y * y) + (z * z))


def normalize_vector(vector: Vector3) -> Vector3:
    norm = vector_norm(vector)
    if norm <= EPSILON:
        return (0.0, 0.0, 0.0)

    x, y, z = vector
    return (x / norm, y / norm, z / norm)


def accel_to_roll_pitch(accel: Vector3) -> tuple[float, float]:
    ax, ay, az = accel
    denominator = math.sqrt((ay * ay) + (az * az))
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, denominator)
    return (roll, pitch)


def quaternion_normalize(quaternion: Quaternion) -> Quaternion:
    w, x, y, z = quaternion
    norm = math.sqrt((w * w) + (x * x) + (y * y) + (z * z))
    if norm <= EPSILON:
        return IDENTITY_QUATERNION
    return (w / norm, x / norm, y / norm, z / norm)


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    w, x, y, z = quaternion
    return (w, -x, -y, -z)


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        (lw * rw) - (lx * rx) - (ly * ry) - (lz * rz),
        (lw * rx) + (lx * rw) + (ly * rz) - (lz * ry),
        (lw * ry) - (lx * rz) + (ly * rw) + (lz * rx),
        (lw * rz) + (lx * ry) - (ly * rx) + (lz * rw),
    )


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Quaternion:
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    return quaternion_normalize(
        (
            (cr * cp * cy) + (sr * sp * sy),
            (sr * cp * cy) - (cr * sp * sy),
            (cr * sp * cy) + (sr * cp * sy),
            (cr * cp * sy) - (sr * sp * cy),
        )
    )


def quaternion_to_euler(quaternion: Quaternion) -> tuple[float, float, float]:
    w, x, y, z = quaternion_normalize(quaternion)

    sinr_cosp = 2.0 * ((w * x) + (y * z))
    cosr_cosp = 1.0 - (2.0 * ((x * x) + (y * y)))
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * ((w * y) - (z * x))
    pitch = math.asin(clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def quaternion_integrate_gyro(
    quaternion: Quaternion,
    gyro_rad_s: Vector3,
    dt_s: float,
) -> Quaternion:
    if dt_s <= EPSILON:
        return quaternion_normalize(quaternion)

    gx, gy, gz = gyro_rad_s
    omega = math.sqrt((gx * gx) + (gy * gy) + (gz * gz))
    if omega <= EPSILON:
        return quaternion_normalize(quaternion)

    half_angle = 0.5 * omega * dt_s
    scale = math.sin(half_angle) / omega
    delta = (
        math.cos(half_angle),
        gx * scale,
        gy * scale,
        gz * scale,
    )
    return quaternion_normalize(quaternion_multiply(quaternion, delta))


def quaternion_lerp(source: Quaternion, target: Quaternion, target_weight: float) -> Quaternion:
    weight = clamp(target_weight, 0.0, 1.0)
    sw, sx, sy, sz = source
    tw, tx, ty, tz = target

    dot = (sw * tw) + (sx * tx) + (sy * ty) + (sz * tz)
    if dot < 0.0:
        tw = -tw
        tx = -tx
        ty = -ty
        tz = -tz

    blended = (
        ((1.0 - weight) * sw) + (weight * tw),
        ((1.0 - weight) * sx) + (weight * tx),
        ((1.0 - weight) * sy) + (weight * ty),
        ((1.0 - weight) * sz) + (weight * tz),
    )
    return quaternion_normalize(blended)


def average_quaternions(quaternions: list[Quaternion]) -> Quaternion:
    if not quaternions:
        return IDENTITY_QUATERNION

    reference = quaternion_normalize(quaternions[0])
    sum_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0

    for quaternion in quaternions:
        w, x, y, z = quaternion_normalize(quaternion)
        dot = (
            (reference[0] * w)
            + (reference[1] * x)
            + (reference[2] * y)
            + (reference[3] * z)
        )
        if dot < 0.0:
            w = -w
            x = -x
            y = -y
            z = -z

        sum_w += w
        sum_x += x
        sum_y += y
        sum_z += z

    return quaternion_normalize((sum_w, sum_x, sum_y, sum_z))


def relative_quaternion(baseline: Quaternion, current: Quaternion) -> Quaternion:
    return quaternion_normalize(
        quaternion_multiply(quaternion_conjugate(quaternion_normalize(baseline)), current)
    )


@dataclass(frozen=True)
class PoseEstimate:
    quaternion: Quaternion
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def pose_estimate_from_quaternion(quaternion: Quaternion) -> PoseEstimate:
    roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(quaternion)
    return PoseEstimate(
        quaternion=quaternion_normalize(quaternion),
        roll_deg=roll_rad * RAD_TO_DEG,
        pitch_deg=pitch_rad * RAD_TO_DEG,
        yaw_deg=yaw_rad * RAD_TO_DEG,
    )


class VectorLowPassFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = clamp(alpha, 0.0, 1.0)
        self._state: dict[str, Vector3] = {}

    @property
    def enabled(self) -> bool:
        return self.alpha < 1.0

    def apply(self, key: str, value: Vector3) -> Vector3:
        current = (float(value[0]), float(value[1]), float(value[2]))
        previous = self._state.get(key)
        if previous is None or self.alpha >= 1.0:
            self._state[key] = current
            return current

        filtered = (
            (self.alpha * current[0]) + ((1.0 - self.alpha) * previous[0]),
            (self.alpha * current[1]) + ((1.0 - self.alpha) * previous[1]),
            (self.alpha * current[2]) + ((1.0 - self.alpha) * previous[2]),
        )
        self._state[key] = filtered
        return filtered


class BasePoseFilter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.quaternion: Quaternion = IDENTITY_QUATERNION
        self.initialized = False

    def reset(self) -> None:
        self.quaternion = IDENTITY_QUATERNION
        self.initialized = False

    def estimate(self) -> PoseEstimate:
        return pose_estimate_from_quaternion(self.quaternion)

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        raise NotImplementedError


class NoFilterPoseFilter(BasePoseFilter):
    def __init__(self) -> None:
        super().__init__("none")

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        gyro_rad_s = tuple(component * DEG_TO_RAD for component in gyro_dps)

        if not self.initialized:
            roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
            self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, 0.0)
            self.initialized = True
            return self.quaternion

        predicted = quaternion_integrate_gyro(self.quaternion, gyro_rad_s, dt_s)
        accel_norm = vector_norm(accel_mps2)
        if accel_norm <= EPSILON:
            self.quaternion = predicted
            return self.quaternion

        roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
        _, _, yaw_rad = quaternion_to_euler(predicted)
        self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
        return self.quaternion


class ComplementaryPoseFilter(BasePoseFilter):
    def __init__(self, alpha: float) -> None:
        super().__init__("complementary")
        self.alpha = clamp(alpha, 0.0, 1.0)

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        gyro_rad_s = tuple(component * DEG_TO_RAD for component in gyro_dps)

        if not self.initialized:
            roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
            self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, 0.0)
            self.initialized = True
            return self.quaternion

        predicted = quaternion_integrate_gyro(self.quaternion, gyro_rad_s, dt_s)
        accel_norm = vector_norm(accel_mps2)
        if accel_norm <= EPSILON:
            self.quaternion = predicted
            return self.quaternion

        roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
        _, _, yaw_rad = quaternion_to_euler(predicted)
        accel_pose = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
        self.quaternion = quaternion_lerp(predicted, accel_pose, 1.0 - self.alpha)
        return self.quaternion


class KalmanAngle1D:
    def __init__(self, q_angle: float, q_bias: float, r_measure: float) -> None:
        self.q_angle = max(q_angle, EPSILON)
        self.q_bias = max(q_bias, EPSILON)
        self.r_measure = max(r_measure, EPSILON)
        self.angle = 0.0
        self.bias = 0.0
        self.p00 = 0.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 0.0

    def reset(self, angle_rad: float = 0.0) -> None:
        self.angle = angle_rad
        self.bias = 0.0
        self.p00 = 0.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 0.0

    def update(self, measured_angle_rad: float, measured_rate_rad_s: float, dt_s: float) -> float:
        rate = measured_rate_rad_s - self.bias
        self.angle += dt_s * rate

        self.p00 += dt_s * ((dt_s * self.p11) - self.p01 - self.p10 + self.q_angle)
        self.p01 -= dt_s * self.p11
        self.p10 -= dt_s * self.p11
        self.p11 += self.q_bias * dt_s

        innovation_covariance = self.p00 + self.r_measure
        kalman_gain_0 = self.p00 / innovation_covariance
        kalman_gain_1 = self.p10 / innovation_covariance

        innovation = wrap_angle_rad(measured_angle_rad - self.angle)
        self.angle += kalman_gain_0 * innovation
        self.bias += kalman_gain_1 * innovation

        p00 = self.p00
        p01 = self.p01

        self.p00 -= kalman_gain_0 * p00
        self.p01 -= kalman_gain_0 * p01
        self.p10 -= kalman_gain_1 * p00
        self.p11 -= kalman_gain_1 * p01

        self.angle = wrap_angle_rad(self.angle)
        return self.angle


class KalmanPoseFilter(BasePoseFilter):
    def __init__(self, q_angle: float, q_bias: float, r_measure: float) -> None:
        super().__init__("kalman")
        self.roll_filter = KalmanAngle1D(q_angle=q_angle, q_bias=q_bias, r_measure=r_measure)
        self.pitch_filter = KalmanAngle1D(q_angle=q_angle, q_bias=q_bias, r_measure=r_measure)
        self.yaw_rad = 0.0

    def reset(self) -> None:
        super().reset()
        self.roll_filter.reset()
        self.pitch_filter.reset()
        self.yaw_rad = 0.0

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
        gyro_rad_s = tuple(component * DEG_TO_RAD for component in gyro_dps)
        gx_rad_s, gy_rad_s, gz_rad_s = gyro_rad_s

        if not self.initialized:
            self.roll_filter.reset(roll_rad)
            self.pitch_filter.reset(pitch_rad)
            self.yaw_rad = 0.0
            self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, self.yaw_rad)
            self.initialized = True
            return self.quaternion

        self.yaw_rad = wrap_angle_rad(self.yaw_rad + (gz_rad_s * dt_s))
        filtered_roll_rad = self.roll_filter.update(roll_rad, gx_rad_s, dt_s)
        filtered_pitch_rad = self.pitch_filter.update(pitch_rad, gy_rad_s, dt_s)
        self.quaternion = quaternion_from_euler(filtered_roll_rad, filtered_pitch_rad, self.yaw_rad)
        return self.quaternion


class MahonyPoseFilter(BasePoseFilter):
    def __init__(self, kp: float, ki: float) -> None:
        super().__init__("mahony")
        self.kp = max(kp, 0.0)
        self.ki = max(ki, 0.0)
        self.integral_error: Vector3 = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        super().reset()
        self.integral_error = (0.0, 0.0, 0.0)

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        gyro_rad_s = tuple(component * DEG_TO_RAD for component in gyro_dps)
        accel_norm = vector_norm(accel_mps2)

        if not self.initialized:
            roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
            self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, 0.0)
            self.initialized = True

        if dt_s <= EPSILON:
            self.quaternion = quaternion_normalize(self.quaternion)
            return self.quaternion

        gx, gy, gz = gyro_rad_s

        if accel_norm > EPSILON:
            ax, ay, az = normalize_vector(accel_mps2)
            qw, qx, qy, qz = quaternion_normalize(self.quaternion)

            # Match the blog's Mahony IMU update form that uses half-vectors and
            # quaternion-derivative integration, while keeping our Euler sign
            # convention unchanged in quaternion_to_euler().
            half_vx = (qx * qz) - (qw * qy)
            half_vy = (qw * qx) + (qy * qz)
            half_vz = (qw * qw) - 0.5 + (qz * qz)

            half_ex = (ay * half_vz) - (az * half_vy)
            half_ey = (az * half_vx) - (ax * half_vz)
            half_ez = (ax * half_vy) - (ay * half_vx)

            if self.ki > 0.0:
                self.integral_error = (
                    self.integral_error[0] + (self.ki * half_ex * dt_s),
                    self.integral_error[1] + (self.ki * half_ey * dt_s),
                    self.integral_error[2] + (self.ki * half_ez * dt_s),
                )
                gx += self.integral_error[0]
                gy += self.integral_error[1]
                gz += self.integral_error[2]
            else:
                self.integral_error = (0.0, 0.0, 0.0)

            gx += self.kp * half_ex
            gy += self.kp * half_ey
            gz += self.kp * half_ez

        qw, qx, qy, qz = quaternion_normalize(self.quaternion)
        half_dt = 0.5 * dt_s
        gx *= half_dt
        gy *= half_dt
        gz *= half_dt

        self.quaternion = quaternion_normalize(
            (
                qw + (-qx * gx) - (qy * gy) - (qz * gz),
                qx + (qw * gx) + (qy * gz) - (qz * gy),
                qy + (qw * gy) - (qx * gz) + (qz * gx),
                qz + (qw * gz) + (qx * gy) - (qy * gx),
            )
        )
        return self.quaternion


class MadgwickPoseFilter(BasePoseFilter):
    def __init__(self, beta: float) -> None:
        super().__init__("madgwick")
        self.beta = max(beta, 0.0)

    def update(self, accel_mps2: Vector3, gyro_dps: Vector3, dt_s: float) -> Quaternion:
        gyro_rad_s = tuple(component * DEG_TO_RAD for component in gyro_dps)
        accel_norm = vector_norm(accel_mps2)

        if not self.initialized:
            roll_rad, pitch_rad = accel_to_roll_pitch(accel_mps2)
            self.quaternion = quaternion_from_euler(roll_rad, pitch_rad, 0.0)
            self.initialized = True

        if dt_s <= EPSILON:
            self.quaternion = quaternion_normalize(self.quaternion)
            return self.quaternion

        if accel_norm <= EPSILON:
            self.quaternion = quaternion_integrate_gyro(self.quaternion, gyro_rad_s, dt_s)
            return self.quaternion

        ax, ay, az = normalize_vector(accel_mps2)
        q1, q2, q3, q4 = quaternion_normalize(self.quaternion)

        two_q1 = 2.0 * q1
        two_q2 = 2.0 * q2
        two_q3 = 2.0 * q3
        two_q4 = 2.0 * q4

        f1 = (two_q2 * q4) - (two_q1 * q3) - ax
        f2 = (two_q1 * q2) + (two_q3 * q4) - ay
        f3 = 1.0 - (two_q2 * q2) - (two_q3 * q3) - az

        j_11_or_24 = two_q3
        j_12_or_23 = two_q4
        j_13_or_22 = two_q1
        j_14_or_21 = two_q2
        j_32 = 2.0 * j_14_or_21
        j_33 = 2.0 * j_11_or_24

        gradient_1 = (j_14_or_21 * f2) - (j_11_or_24 * f1)
        gradient_2 = (j_12_or_23 * f1) + (j_13_or_22 * f2) - (j_32 * f3)
        gradient_3 = (j_12_or_23 * f2) - (j_33 * f3) - (j_13_or_22 * f1)
        gradient_4 = (j_14_or_21 * f1) + (j_11_or_24 * f2)

        gradient_norm = math.sqrt(
            (gradient_1 * gradient_1)
            + (gradient_2 * gradient_2)
            + (gradient_3 * gradient_3)
            + (gradient_4 * gradient_4)
        )
        if gradient_norm > EPSILON:
            # The textbook implementation normalizes the gradient every update.
            # With this project's slow sample rate, that creates a near-constant
            # correction step and can cause limit-cycle oscillation around the
            # settled pose. Keep large-error behavior the same, but let the
            # correction taper naturally once the gradient is already small.
            if gradient_norm > 1.0:
                gradient_1 /= gradient_norm
                gradient_2 /= gradient_norm
                gradient_3 /= gradient_norm
                gradient_4 /= gradient_norm
        else:
            gradient_1 = 0.0
            gradient_2 = 0.0
            gradient_3 = 0.0
            gradient_4 = 0.0

        gx_rad_s, gy_rad_s, gz_rad_s = gyro_rad_s
        q_dot_1 = (-0.5 * q2 * gx_rad_s) - (0.5 * q3 * gy_rad_s) - (0.5 * q4 * gz_rad_s)
        q_dot_2 = (0.5 * q1 * gx_rad_s) + (0.5 * q3 * gz_rad_s) - (0.5 * q4 * gy_rad_s)
        q_dot_3 = (0.5 * q1 * gy_rad_s) - (0.5 * q2 * gz_rad_s) + (0.5 * q4 * gx_rad_s)
        q_dot_4 = (0.5 * q1 * gz_rad_s) + (0.5 * q2 * gy_rad_s) - (0.5 * q3 * gx_rad_s)

        q1 += (q_dot_1 - (self.beta * gradient_1)) * dt_s
        q2 += (q_dot_2 - (self.beta * gradient_2)) * dt_s
        q3 += (q_dot_3 - (self.beta * gradient_3)) * dt_s
        q4 += (q_dot_4 - (self.beta * gradient_4)) * dt_s

        self.quaternion = quaternion_normalize((q1, q2, q3, q4))
        return self.quaternion


def build_pose_filter(
    name: str,
    *,
    complementary_alpha: float,
    kalman_q_angle: float,
    kalman_q_bias: float,
    kalman_r_measure: float,
    mahony_kp: float,
    mahony_ki: float,
    madgwick_beta: float,
) -> BasePoseFilter:
    normalized_name = name.strip().lower()
    if normalized_name == "none":
        return NoFilterPoseFilter()
    if normalized_name == "complementary":
        return ComplementaryPoseFilter(alpha=complementary_alpha)
    if normalized_name == "kalman":
        return KalmanPoseFilter(
            q_angle=kalman_q_angle,
            q_bias=kalman_q_bias,
            r_measure=kalman_r_measure,
        )
    if normalized_name == "mahony":
        return MahonyPoseFilter(kp=mahony_kp, ki=mahony_ki)
    if normalized_name == "madgwick":
        return MadgwickPoseFilter(beta=madgwick_beta)

    raise ValueError(f"Unsupported filter name: {name}")
