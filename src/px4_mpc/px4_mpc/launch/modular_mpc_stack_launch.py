import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('px4_mpc')
    rviz_config_path = os.path.join(pkg_dir, 'config', 'visualize.rviz')

    dds_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent udp4 -p 8888'],
        shell=True,
        output='screen'
    )

    brain_node = Node(
        package='px4_mpc',
        executable='decision_brain',
        name='decision_brain_node',
        output='screen'
    )

    controller_node = Node(
        package='px4_mpc',
        executable='mpc_controller',
        name='modular_mpc_controller_node',
        output='screen'
    )

    detection_node = Node(
        package='px4_mpc',
        executable='pipeline_detection',
        name='pipeline_detection_node',
        output='screen'
    )

    transport_node = Node(
        package='px4_mpc',
        executable='mission_transport',
        name='mission_transport_node',
        output='screen'
    )

    visualizer_node = Node(
        package='px4_mpc',
        executable='rviz_visualizer',
        name='rviz_visualizer_node',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
        output='screen'
    )

    return LaunchDescription([
        dds_agent,
        brain_node,
        controller_node,
        detection_node,
        transport_node,
        visualizer_node,
        rviz_node,
    ])
