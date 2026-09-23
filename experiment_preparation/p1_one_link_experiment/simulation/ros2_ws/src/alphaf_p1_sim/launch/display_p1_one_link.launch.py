"""Display the P1 one-link model in RViz when GUI packages are available."""

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
    rviz_path = os.path.join(pkg_share, "rviz", "so101_p1_one_link.rviz")
    robot_description = {"robot_description": xacro.process_file(xacro_path).toxml()}

    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),
        Node(
            package="alphaf_p1_sim",
            executable="publish_p1_target_markers.py",
            name="publish_p1_target_markers",
            output="screen",
        ),
    ]
    if _package_exists("joint_state_publisher_gui"):
        actions.append(
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
            )
        )
    else:
        actions.extend(
            [
                LogInfo(msg="joint_state_publisher_gui package not found; using p1_joint_state_test fallback"),
                Node(
                    package="alphaf_p1_sim",
                    executable="p1_joint_state_test.py",
                    name="p1_joint_state_test",
                    output="screen",
                ),
            ]
        )
    if _package_exists("rviz2"):
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_path],
            )
        )
    else:
        actions.append(LogInfo(msg="rviz2 package not found; skipping RViz display"))
    return LaunchDescription(actions)
