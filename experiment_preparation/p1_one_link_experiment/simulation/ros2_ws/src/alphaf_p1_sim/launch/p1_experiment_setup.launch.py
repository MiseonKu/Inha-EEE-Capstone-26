"""Simulation-only P1 experiment setup with model and target markers."""

from __future__ import annotations

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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

    use_rviz = LaunchConfiguration("use_rviz")
    joint_source = LaunchConfiguration("joint_source")
    targets_csv = LaunchConfiguration("targets_csv")
    trajectories_csv = LaunchConfiguration("trajectories_csv")
    player_loop = LaunchConfiguration("player_loop")
    use_vive_comparator = LaunchConfiguration("use_vive_comparator")
    vive_pose_topic = LaunchConfiguration("vive_pose_topic")
    calibration_json = LaunchConfiguration("calibration_json")
    comparison_log = LaunchConfiguration("comparison_log")

    actions = [
        DeclareLaunchArgument("use_rviz", default_value="false", description="Open RViz if rviz2 is available"),
        DeclareLaunchArgument("joint_source", default_value="test", description="test or trajectory"),
        DeclareLaunchArgument("targets_csv", default_value="", description="Optional target CSV path"),
        DeclareLaunchArgument("trajectories_csv", default_value="", description="Optional trajectory CSV path"),
        DeclareLaunchArgument("player_loop", default_value="false", description="Loop trajectory playback"),
        DeclareLaunchArgument("use_vive_comparator", default_value="false", description="Subscribe to Vive pose and compare against active target"),
        DeclareLaunchArgument("vive_pose_topic", default_value="/vive_tracker/pose", description="PoseStamped topic for Vive Tracker"),
        DeclareLaunchArgument("calibration_json", default_value="", description="Vive-world to base_link calibration JSON"),
        DeclareLaunchArgument("comparison_log", default_value="", description="Optional live comparison CSV log"),
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
            parameters=[{"targets_csv": targets_csv}],
        ),
        Node(
            package="alphaf_p1_sim",
            executable="p1_joint_state_test.py",
            name="p1_joint_state_test",
            output="screen",
            condition=IfCondition(PythonExpression(["'", joint_source, "' == 'test'"])),
        ),
        Node(
            package="alphaf_p1_sim",
            executable="p1_trajectory_player.py",
            name="p1_trajectory_player",
            output="screen",
            parameters=[{"trajectories_csv": trajectories_csv, "loop": ParameterValue(player_loop, value_type=bool)}],
            condition=IfCondition(PythonExpression(["'", joint_source, "' == 'trajectory'"])),
        ),
        Node(
            package="alphaf_p1_sim",
            executable="p1_vive_target_comparator.py",
            name="p1_vive_target_comparator",
            output="screen",
            parameters=[
                {
                    "targets_csv": targets_csv,
                    "vive_pose_topic": vive_pose_topic,
                    "calibration_json": calibration_json,
                    "output_log": comparison_log,
                }
            ],
            condition=IfCondition(use_vive_comparator),
        ),
    ]

    if _package_exists("rviz2"):
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_path],
                condition=IfCondition(use_rviz),
            )
        )
    else:
        actions.append(LogInfo(msg="rviz2 package not found; use_rviz cannot open RViz"))
    return LaunchDescription(actions)
