#!/usr/bin/env python3
"""Play generated P1 trajectories as JointState messages for simulation only."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32


def _find_default_trajectories() -> str:
    env_path = os.environ.get("ALPHAF_P1_TRAJECTORIES_CSV")
    if env_path:
        return env_path
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        candidate = base / "outputs" / "pre_imu_p1" / "p1_trajectories.csv"
        if candidate.exists():
            return str(candidate)
    return ""


class P1TrajectoryPlayer(Node):
    """Replay generated command trajectories on /joint_states."""

    def __init__(self) -> None:
        super().__init__("p1_trajectory_player")
        self.declare_parameter("trajectories_csv", _find_default_trajectories())
        self.declare_parameter("loop", True)
        self.rows = self._load_rows()
        self.index = 0
        self.loop = self.get_parameter("loop").get_parameter_value().bool_value
        self.timer = self.create_timer(0.02, self._publish)
        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.target_id_publisher = self.create_publisher(Int32, "p1_active_target_id", 10)
        self.trial_id_publisher = self.create_publisher(Int32, "p1_active_trial_id", 10)
        self.get_logger().info(f"loaded {len(self.rows)} P1 trajectory samples")

    def _load_rows(self) -> list[dict[str, float]]:
        path_text = self.get_parameter("trajectories_csv").get_parameter_value().string_value
        if not path_text:
            self.get_logger().warning("no trajectories_csv path provided or discovered")
            return []
        path = Path(path_text)
        if not path.exists():
            self.get_logger().warning(f"trajectories_csv does not exist: {path}")
            return []
        rows: list[dict[str, float]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "trial_id": int(row["trial_id"]),
                        "target_id": int(row["target_id"]),
                        "q1_rad": float(row["q1_rad"]),
                        "q2_rad": float(row["q2_rad"]),
                    }
                )
        return rows

    def _publish(self) -> None:
        if not self.rows:
            return
        if self.index >= len(self.rows):
            if not self.loop:
                return
            self.index = 0
        row = self.rows[self.index]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint_1_yaw", "joint_2_pitch"]
        msg.position = [row["q1_rad"], row["q2_rad"]]
        self.publisher.publish(msg)
        target_msg = Int32()
        target_msg.data = int(row["target_id"])
        self.target_id_publisher.publish(target_msg)
        trial_msg = Int32()
        trial_msg.data = int(row["trial_id"])
        self.trial_id_publisher.publish(trial_msg)
        self.index += 1


def main() -> None:
    rclpy.init()
    node = P1TrajectoryPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
