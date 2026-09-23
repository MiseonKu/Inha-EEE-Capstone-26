cmake_minimum_required(VERSION 3.10)
project(alphaf_p1_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/publish_p1_target_markers.py
    scripts/p1_joint_state_test.py
    scripts/p1_trajectory_player.py
    scripts/p1_vive_target_comparator.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
