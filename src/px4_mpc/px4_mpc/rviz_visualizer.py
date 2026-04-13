#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from visualization_msgs.msg import Marker

from mpc_msgs.msg import GasConcentration

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
        self.gas_concentration_sub = self.create_subscription(
            GasConcentration, '/perception/gas_concentration',
            self.gas_concentration_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/mpc/path', 10)
        self.gas_leak_marker_pub = self.create_publisher(Marker, '/mpc/gas_leak_marker', 10)
        self.gas_status_marker_pub = self.create_publisher(Marker, '/mpc/gas_status_marker', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Path Data
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.leak_position = None
        self.current_gas_concentration = 0.0

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

    def gas_concentration_callback(self, msg):
        self.leak_position = np.array([
            msg.leak_position.x,
            msg.leak_position.y,
            msg.leak_position.z
        ], dtype=float)
        self.current_gas_concentration = float(msg.concentration)

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
        self.publish_gas_markers(now, rel_pos)

    def publish_gas_markers(self, now, rel_pos):
        if self.leak_position is None:
            return

        leak_rel_pos = self.leak_position - self.map_origin

        leak_marker = Marker()
        leak_marker.header.stamp = now
        leak_marker.header.frame_id = "map"
        leak_marker.ns = "gas_leak"
        leak_marker.id = 0
        leak_marker.type = Marker.SPHERE
        leak_marker.action = Marker.ADD
        leak_marker.pose.position.x = float(leak_rel_pos[0])
        leak_marker.pose.position.y = float(leak_rel_pos[1])
        leak_marker.pose.position.z = float(leak_rel_pos[2])
        leak_marker.pose.orientation.w = 1.0
        leak_marker.scale.x = 0.35
        leak_marker.scale.y = 0.35
        leak_marker.scale.z = 0.35
        leak_marker.color.a = 0.85
        leak_marker.color.r = 1.0
        leak_marker.color.g = 0.45
        leak_marker.color.b = 0.05
        self.gas_leak_marker_pub.publish(leak_marker)

        status_marker = Marker()
        status_marker.header.stamp = now
        status_marker.header.frame_id = "map"
        status_marker.ns = "gas_status"
        status_marker.id = 1
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.pose.position.x = float(rel_pos[0])
        status_marker.pose.position.y = float(rel_pos[1])
        status_marker.pose.position.z = float(rel_pos[2] + 0.8)
        status_marker.pose.orientation.w = 1.0
        status_marker.scale.z = 0.3
        status_marker.color.a = 1.0
        status_marker.color.r = 0.2
        status_marker.color.g = 0.95
        status_marker.color.b = 0.25
        status_marker.text = f"gas: {self.current_gas_concentration:.3f}"
        self.gas_status_marker_pub.publish(status_marker)

def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
