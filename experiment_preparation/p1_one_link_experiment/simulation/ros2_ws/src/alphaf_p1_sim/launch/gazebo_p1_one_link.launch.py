"""Launch Gazebo/GZ simulation for the P1 one-link model when available."""

from __future__ import annotations

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def _share(name: str) -> str | None:
    try:
        return get_package_share_directory(name)
    except PackageNotFoundError:
        return None


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
        ),
        Node(
            package="alphaf_p1_sim",
            executable="p1_joint_state_test.py",
            name="p1_joint_state_test",
            output="screen",
        ),
    ]

    gazebo_ros_share = _share("gazebo_ros")
    ros_gz_share = _share("ros_gz_sim")
    if gazebo_ros_share:
        actions.extend(
            [
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py"))
                ),
                Node(
                    package="gazebo_ros",
                    executable="spawn_entity.py",
                    arguments=["-topic", "robot_description", "-entity", "alphaf_p1_one_link"],
                    output="screen",
                ),
            ]
        )
    elif ros_gz_share:
        actions.extend(
            [
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(ros_gz_share, "launch", "gz_sim.launch.py")),
                    launch_arguments={"gz_args": "-r empty.sdf"}.items(),
                ),
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    arguments=["-name", "alphaf_p1_one_link", "-topic", "robot_description"],
                    output="screen",
                ),
            ]
        )
    else:
        actions.append(LogInfo(msg="Neither gazebo_ros nor ros_gz_sim is available; Gazebo spawn skipped"))
    return LaunchDescription(actions)
