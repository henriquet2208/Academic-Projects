import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Launch Argument Configurations
    world_path = LaunchConfiguration('world')
    name = LaunchConfiguration('name')
    namespace = LaunchConfiguration('namespace')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')
    r_orientation = LaunchConfiguration('r_orientation')
    p_orientation = LaunchConfiguration('p_orientation')
    y_orientation = LaunchConfiguration('y_orientation')

    # Launch Arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='/home/iacos/project_iacos/worlds/gazebo.world',
        description='Path to the Gazebo world file'
    )
    name_arg = DeclareLaunchArgument(
        'name',
        default_value='prius',
        description='Gazebo robot object name'
    )
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='prius',
        description='ROS2 robot namespace'
    )
    x_pose_arg = DeclareLaunchArgument(
        'x_pose',
        default_value='0.0',
        description='Robot spawn x position'
    )
    y_pose_arg = DeclareLaunchArgument(
        'y_pose',
        default_value='0.0',
        description='Robot spawn y position'
    )
    z_pose_arg = DeclareLaunchArgument(
        'z_pose',
        default_value='0.0',
        description='Robot spawn z position'
    )
    r_orientation_arg = DeclareLaunchArgument(
        'r_orientation',
        default_value='0.0',
        description='Robot spawn roll orientation angle'
    )
    p_orientation_arg = DeclareLaunchArgument(
        'p_orientation',
        default_value='0.0',
        description='Robot spawn pitch orientation angle'
    )
    y_orientation_arg = DeclareLaunchArgument(
        'y_orientation',
        default_value='0.0',
        description='Robot spawn yaw orientation angle'
    )

    # Paths
    gazebo_ros_pkg_share = get_package_share_directory('gazebo_ros')
    prius_description_pkg_share = get_package_share_directory('prius_description')
    urdf_file_path = os.path.join(prius_description_pkg_share, 'urdf', 'prius.urdf')

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([gazebo_ros_pkg_share, 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={'world': world_path}.items()
    )

    # Spawn Prius node
    spawn_prius = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_prius',
        namespace=namespace,
        output='screen',
        arguments=[
            '-entity', name,
            '-file', urdf_file_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
            '-R', r_orientation,
            '-P', p_orientation,
            '-Y', y_orientation
        ]
    )

    # Launch description
    return LaunchDescription([
        world_arg,
        name_arg,
        namespace_arg,
        x_pose_arg,
        y_pose_arg,
        z_pose_arg,
        r_orientation_arg,
        p_orientation_arg,
        y_orientation_arg,
        gazebo_launch,
        spawn_prius,
    ])
