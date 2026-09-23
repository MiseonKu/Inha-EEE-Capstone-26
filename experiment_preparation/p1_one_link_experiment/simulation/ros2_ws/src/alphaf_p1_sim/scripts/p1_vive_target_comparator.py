#!/usr/bin/env python3
"""Compare live Vive Tracker pose against the active P1 target.

This node does not command hardware. It subscribes to a Vive PoseStamped topic,
optionally applies a Vive-world to base_link calibration transform, listens to
the active target id, and publishes/logs the target-vs-measured error.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Point
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray


LOG_COLUMNS = [
    "stamp_s",
    "trial_id",
    "target_id",
    "q1_encoder_rad",
    "q2_encoder_rad",
    "p_target_base_x_m",
    "p_target_base_y_m",
    "p_target_base_z_m",
    "p_vive_world_x_m",
    "p_vive_world_y_m",
    "p_vive_world_z_m",
    "p_vive_base_x_m",
    "p_vive_base_y_m",
    "p_vive_base_z_m",
    "error_x_m",
    "error_y_m",
    "error_z_m",
    "error_norm_m",
]


def _read_targets(path: Path) -> dict[int, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        int(row["target_id"]): np.array(
            [
                float(row["p_target_base_x_m"]),
                float(row["p_target_base_y_m"]),
                float(row["p_target_base_z_m"]),
            ],
            dtype=float,
        )
        for row in rows
    }


def _load_transform(path_text: str) -> tuple[np.ndarray, np.ndarray, str]:
    if not path_text:
        return np.eye(3), np.zeros(3), "identity_no_calibration_file"
    path = Path(path_text)
    if not path.exists():
        return np.eye(3), np.zeros(3), f"identity_missing_calibration_file:{path}"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    R = np.asarray(data["R_base_from_vive_world"], dtype=float).reshape(3, 3)
    t = np.asarray(data["t_base_from_vive_world_m"], dtype=float).reshape(3)
    return R, t, str(path)


class P1ViveTargetComparator(Node):
    """Live target-vs-Vive comparator and logger."""

    def __init__(self) -> None:
        super().__init__("p1_vive_target_comparator")
        self.declare_parameter("targets_csv", "")
        self.declare_parameter("calibration_json", "")
        self.declare_parameter("vive_pose_topic", "/vive_tracker/pose")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("active_target_topic", "/p1_active_target_id")
        self.declare_parameter("active_trial_topic", "/p1_active_trial_id")
        self.declare_parameter("marker_topic", "/p1_vive_comparison")
        self.declare_parameter("output_log", "")
        self.declare_parameter("frame_id", "base_link")

        targets_path = Path(self.get_parameter("targets_csv").value)
        if not targets_path.exists():
            raise FileNotFoundError(f"targets_csv is required and must exist: {targets_path}")
        self.targets = _read_targets(targets_path)
        self.active_target_id = min(self.targets)
        self.active_trial_id = -1
        self.latest_q = {"joint_1_yaw": np.nan, "joint_2_pitch": np.nan}

        self.R, self.t, transform_source = _load_transform(str(self.get_parameter("calibration_json").value))
        if transform_source.startswith("identity"):
            self.get_logger().warning(f"using identity Vive->base transform: {transform_source}")
        else:
            self.get_logger().info(f"loaded Vive->base calibration: {transform_source}")

        self.output_log = Path(str(self.get_parameter("output_log").value)) if self.get_parameter("output_log").value else None
        if self.output_log:
            self.output_log.parent.mkdir(parents=True, exist_ok=True)
            if not self.output_log.exists():
                with self.output_log.open("w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()

        self.marker_pub = self.create_publisher(MarkerArray, str(self.get_parameter("marker_topic").value), 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("vive_pose_topic").value), self._on_vive_pose, 20)
        self.create_subscription(JointState, str(self.get_parameter("joint_states_topic").value), self._on_joint_state, 20)
        self.create_subscription(Int32, str(self.get_parameter("active_target_topic").value), self._on_target_id, 20)
        self.create_subscription(Int32, str(self.get_parameter("active_trial_topic").value), self._on_trial_id, 20)
        self.get_logger().info(f"loaded {len(self.targets)} targets for live Vive comparison")

    def _on_target_id(self, msg: Int32) -> None:
        if int(msg.data) in self.targets:
            self.active_target_id = int(msg.data)

    def _on_trial_id(self, msg: Int32) -> None:
        self.active_trial_id = int(msg.data)

    def _on_joint_state(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            if name in self.latest_q:
                self.latest_q[name] = float(position)

    def _on_vive_pose(self, msg: PoseStamped) -> None:
        p_world = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        p_base = self.R @ p_world + self.t
        target = self.targets[self.active_target_id]
        error = p_base - target
        error_norm = float(np.linalg.norm(error))
        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp_s <= 0.0:
            stamp_s = self.get_clock().now().nanoseconds * 1e-9

        row = {
            "stamp_s": stamp_s,
            "trial_id": self.active_trial_id,
            "target_id": self.active_target_id,
            "q1_encoder_rad": self.latest_q["joint_1_yaw"],
            "q2_encoder_rad": self.latest_q["joint_2_pitch"],
            "p_target_base_x_m": target[0],
            "p_target_base_y_m": target[1],
            "p_target_base_z_m": target[2],
            "p_vive_world_x_m": p_world[0],
            "p_vive_world_y_m": p_world[1],
            "p_vive_world_z_m": p_world[2],
            "p_vive_base_x_m": p_base[0],
            "p_vive_base_y_m": p_base[1],
            "p_vive_base_z_m": p_base[2],
            "error_x_m": error[0],
            "error_y_m": error[1],
            "error_z_m": error[2],
            "error_norm_m": error_norm,
        }
        self._append_log(row)
        self.marker_pub.publish(self._markers(target, p_base, error_norm))

    def _append_log(self, row: dict[str, object]) -> None:
        if not self.output_log:
            return
        with self.output_log.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerow(row)

    def _markers(self, target: np.ndarray, vive: np.ndarray, error_norm: float) -> MarkerArray:
        frame_id = str(self.get_parameter("frame_id").value)
        stamp = self.get_clock().now().to_msg()
        target_marker = Marker()
        target_marker.header.frame_id = frame_id
        target_marker.header.stamp = stamp
        target_marker.ns = "p1_live_target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = float(target[0])
        target_marker.pose.position.y = float(target[1])
        target_marker.pose.position.z = float(target[2])
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.025
        target_marker.color.g = 1.0
        target_marker.color.a = 1.0

        vive_marker = Marker()
        vive_marker.header.frame_id = frame_id
        vive_marker.header.stamp = stamp
        vive_marker.ns = "p1_live_vive"
        vive_marker.id = 1
        vive_marker.type = Marker.SPHERE
        vive_marker.action = Marker.ADD
        vive_marker.pose.position.x = float(vive[0])
        vive_marker.pose.position.y = float(vive[1])
        vive_marker.pose.position.z = float(vive[2])
        vive_marker.pose.orientation.w = 1.0
        vive_marker.scale.x = vive_marker.scale.y = vive_marker.scale.z = 0.025
        vive_marker.color.r = 0.85
        vive_marker.color.b = 1.0
        vive_marker.color.a = 1.0

        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = "p1_live_error"
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.points = [Point(x=float(target[0]), y=float(target[1]), z=float(target[2])), Point(x=float(vive[0]), y=float(vive[1]), z=float(vive[2]))]
        line.scale.x = 0.006
        line.color.r = 1.0
        line.color.g = 0.05 if error_norm > 0.01 else 0.7
        line.color.a = 1.0
        return MarkerArray(markers=[target_marker, vive_marker, line])


def main() -> None:
    rclpy.init()
    node = P1ViveTargetComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
