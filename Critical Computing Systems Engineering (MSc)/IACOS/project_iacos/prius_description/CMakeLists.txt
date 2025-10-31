cmake_minimum_required(VERSION 3.5)
project(prius_description)
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(
  DIRECTORY meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

install(FILES model.config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()