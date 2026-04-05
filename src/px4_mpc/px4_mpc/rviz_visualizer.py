#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class RVizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer_node')

        # 👑 地图原点偏移：设定为航点 [-1.0, -1.0]，但 Z 设为 0
        self.map_origin = np.array([-1.0, -1.0, 0.0])

        # QoS profile for PX4 topics
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', 
            self.vehicle_local_position_callback, qos_profile)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', 
            self.vehicle_attitude_callback, qos_profile)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/mpc/path', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Path Data
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])

        # Timer for visualization update (20Hz is enough for RViz)
        self.create_timer(0.05, self.visual_timer_callback)

    def vehicle_local_position_callback(self, msg):
        # Convert PX4 NED to ENU for RViz
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z

    def vehicle_attitude_callback(self, msg):
        # Convert PX4 NED to ENU for RViz
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

    def visual_timer_callback(self):
        now = self.get_clock().now().to_msg()
        
        # 计算相对于地图原点的坐标
        rel_pos = self.vehicle_local_position - self.map_origin
        
        # 1. 发布 TF (map -> base_link)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(rel_pos[0])
        t.transform.translation.y = float(rel_pos[1])
        t.transform.translation.z = float(rel_pos[2])
        t.transform.rotation.w = float(self.vehicle_attitude[0])
        t.transform.rotation.x = float(self.vehicle_attitude[1])
        t.transform.rotation.y = float(self.vehicle_attitude[2])
        t.transform.rotation.z = float(self.vehicle_attitude[3])
        self.tf_broadcaster.sendTransform(t)

        # 2. 更新并发布 Path
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "map"
        pose.pose.position.x = float(rel_pos[0])
        pose.pose.position.y = float(rel_pos[1])
        pose.pose.position.z = float(rel_pos[2])
        
        # 👑 解决路径太短的问题：增加缓存上限到 5000 个点
        self.path_msg.header.stamp = now
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses.pop(0)
            
        self.path_pub.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
