cmake_minimum_required(VERSION 3.8)
project(data_sync)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Register custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SyncedOutput.msg"
  DEPENDENCIES builtin_interfaces sensor_msgs std_msgs
)

# Install Python scripts (not a module)
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_package()
