cmake_minimum_required(VERSION 3.8)
project(dual_t265_stitching)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(realsense2 REQUIRED)
find_package(ament_index_cpp REQUIRED)


# === C++ Executable: dual_t265_node ===
add_executable(dual_t265_node src/dual_t265_node.cpp)
ament_target_dependencies(dual_t265_node
  rclcpp
  nav_msgs
  sensor_msgs
  tf2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  cv_bridge
  OpenCV
  realsense2
)

target_link_libraries(dual_t265_node
  ${OpenCV_LIBS}
  ${cv_bridge_LIBRARIES}
  ${realsense2_LIBRARY}
)

# === Install C++ executables ===
install(TARGETS
  dual_t265_node
  DESTINATION lib/${PROJECT_NAME}
)
# === Install Python executables ===
install(PROGRAMS
  scripts/dual_fisheye_stitcher.py
  scripts/stitcher_node.py
  scripts/overlap_calibration.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
