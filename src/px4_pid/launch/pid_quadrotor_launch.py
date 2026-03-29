from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='px4_pid',
            executable='pid_quadrotor',
            name='pid_quadrotor_node',
            output='screen'
        ),
    ])
