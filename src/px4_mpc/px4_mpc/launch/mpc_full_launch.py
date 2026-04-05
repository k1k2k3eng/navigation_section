import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包路径
    pkg_dir = get_package_share_directory('px4_mpc')
    
    # 0. Micro-XRCE-DDS-Agent
    # 一键启动代理，这是和 PX4 通信的桥梁
    dds_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent udp4 -p 8888'],
        shell=True,
        output='screen'
    )
    
    # 1. MPC 控制节点
    mpc_node = Node(
        package='px4_mpc',
        executable='mpc_quadrotor',
        name='mpc_quadrotor_node',
        output='screen'
    )

    # 2. 可视化节点 (TF + Path)
    visualizer_node = Node(
        package='px4_mpc',
        executable='rviz_visualizer',
        name='rviz_visualizer_node',
        output='screen'
    )

    # 3. RViz2 节点
    # 加载保存在 config 目录下的可视化配置
    rviz_config_path = os.path.join(pkg_dir, 'config', 'visualize.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path] if os.path.exists(rviz_config_path) else [],
        output='screen'
    )

    return LaunchDescription([
        dds_agent,
        mpc_node,
        visualizer_node,
        rviz_node
    ])
