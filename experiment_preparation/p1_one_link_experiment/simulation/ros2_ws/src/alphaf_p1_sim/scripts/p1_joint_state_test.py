#!/usr/bin/env python3
"""Publish P1 FK validation joint states for simulation-only visualization."""

from __future__ import annotations

from math import pi

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class P1JointStateTest(Node):
    """Cycle through the three required FK validation poses."""

    def __init__(self) -> None:
        super().__init__("p1_joint_state_test")
        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.cases = [
            ("q1=0 q2=0", [0.0, 0.0]),
            ("q1=0 q2=pi/2", [0.0, pi / 2.0]),
            ("q1=pi/2 q2=0", [pi / 2.0, 0.0]),
        ]
        self.index = 0
        self.timer = self.create_timer(2.0, self._publish)
        self.get_logger().info("publishing P1 simulation-only joint-state validation sequence")

    def _publish(self) -> None:
        label, q = self.cases[self.index % len(self.cases)]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint_1_yaw", "joint_2_pitch"]
        msg.position = q
        self.publisher.publish(msg)
        self.get_logger().info(label, throttle_duration_sec=6.0)
        self.index += 1


def main() -> None:
    rclpy.init()
    node = P1JointStateTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
