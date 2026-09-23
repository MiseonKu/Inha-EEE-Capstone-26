"""Launch robot_state_publisher for the P1 one-link model."""

from __future__ import annotations

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
import xacro


def _package_exists(name: str) -> bool:
    try:
        get_package_share_directory(name)
        return True
    except PackageNotFoundError:
        return False


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("alphaf_p1_sim")
    xacro_path = os.path.join(pkg_share, "urdf", "so101_p1_one_link.xacro")
    robot_description = {"robot_description": xacro.process_file(xacro_path).toxml()}

    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        )
    ]
    if _package_exists("joint_state_publisher"):
        actions.append(
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                output="screen",
            )
        )
    else:
        actions.extend(
            [
                LogInfo(msg="joint_state_publisher package not found; using alphaf_p1_sim p1_joint_state_test fallback"),
                Node(
                    package="alphaf_p1_sim",
                    executable="p1_joint_state_test.py",
                    name="p1_joint_state_test",
                    output="screen",
                ),
            ]
        )
    return LaunchDescription(actions)
