cmake_minimum_required(VERSION 3.5)
project(project_iacos)

# Find ament packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(prius_description REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install Python scripts
install(PROGRAMS
  scripts/your_python_script.py  # Replace with actual script names
  DESTINATION lib/${PROJECT_NAME}
)

# Declare this package as ament_python if you are using Python
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install URDF and other resources
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}/urdf
)

# Install other resources (e.g., meshes)
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
