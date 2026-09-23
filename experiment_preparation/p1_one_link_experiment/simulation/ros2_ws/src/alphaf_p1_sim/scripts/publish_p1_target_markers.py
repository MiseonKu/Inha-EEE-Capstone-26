#!/usr/bin/env python3
"""Publish P1 target points as RViz MarkerArray."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


def _find_default_targets() -> str:
    env_path = os.environ.get("ALPHAF_P1_TARGETS_CSV")
    if env_path:
        return env_path
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        candidate = base / "outputs" / "pre_imu_p1" / "p1_targets_pilot.csv"
        if candidate.exists():
            return str(candidate)
    return ""


class P1TargetMarkerPublisher(Node):
    """Publish target marker spheres in base_link."""

    def __init__(self) -> None:
        super().__init__("publish_p1_target_markers")
        self.declare_parameter("targets_csv", _find_default_targets())
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("topic", "p1_targets")
        self.declare_parameter("timer_period_s", 1.0)
        topic = self.get_parameter("topic").get_parameter_value().string_value
        period = self.get_parameter("timer_period_s").get_parameter_value().double_value
        self.publisher = self.create_publisher(MarkerArray, topic, 1)
        self.markers = self._load_markers()
        self.timer = self.create_timer(max(period, 0.1), self._publish)
        self.get_logger().info(f"loaded {len(self.markers.markers)} P1 target markers")

    def _load_markers(self) -> MarkerArray:
        path_text = self.get_parameter("targets_csv").get_parameter_value().string_value
        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        marker_array = MarkerArray()
        if not path_text:
            self.get_logger().warning("no targets_csv path provided or discovered")
            return marker_array
        path = Path(path_text)
        if not path.exists():
            self.get_logger().warning(f"targets_csv does not exist: {path}")
            return marker_array

        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.ns = "p1_targets"
                marker.id = int(row["target_id"])
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(row["p_target_base_x_m"])
                marker.pose.position.y = float(row["p_target_base_y_m"])
                marker.pose.position.z = float(row["p_target_base_z_m"])
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.020
                marker.scale.y = 0.020
                marker.scale.z = 0.020
                marker.color.r = 1.0
                marker.color.g = 0.45
                marker.color.b = 0.05
                marker.color.a = 0.9
                marker_array.markers.append(marker)
        return marker_array

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for marker in self.markers.markers:
            marker.header.stamp = stamp
        self.publisher.publish(self.markers)


def main() -> None:
    rclpy.init()
    node = P1TargetMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
