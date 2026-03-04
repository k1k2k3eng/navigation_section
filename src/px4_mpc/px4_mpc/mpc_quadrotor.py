#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint

from mpc_msgs.srv import SetPose

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC

class QuadrotorMPC(Node):

    def __init__(self):
        super().__init__('quadrotor_mpc_node') # 小菲帮你改了个帅气的名字喵

        # --- 喵！这里是小菲帮你加的航点管理部分 ---
        # 注意！小菲把你的 Z = -1.0 全部改成了 Z = 1.0 (ENU 坐标系，正数才是升空喵！)
        self.waypoints = [[0.0, 0.0, 1.0, 1.57],[-1.0, -1.0, 1.0, 1.57],[5.0, -1.0, 1.0, 1.57],[5.0,3.5, 1.0, 0.0],[0.5, 3.5, 1.0, -1.57],[0.5, 2.0, 1.0, -3.14],[2.0, 2.0, 1.0, 1.57],[2.0, 0.5, 1.0, -3.14],[-1.0, 0.5, 1.0, -1.57]
        ]
        self.current_wp_index = 0
        self.acceptance_radius = 0.3
        self.is_mission_finished = False
        # ------------------------------------------

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.vehicle_status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile_sub)
        self.attitude_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile_sub)
        self.local_position_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile_sub)

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile_pub)
        self.predicted_path_pub = self.create_publisher(Path, '/px4_mpc/predicted_path', 10)
        self.reference_pub = self.create_publisher(Marker, '/px4_mpc/reference', 10)

        # Timer (50Hz)
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state =VehicleStatus.NAVIGATION_STATE_MANUAL # 初始状态设为手动喵！

        # Create MPC
        self.model = MultirotorRateModel()
        MPC_HORIZON = 15
        self.mpc = MultirotorRateMPC(self.model)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        
        # 初始目标点设为第一个航点喵！
        self.setpoint_position = np.array([self.waypoints[0][0], self.waypoints[0][1], self.waypoints[0][2]])
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.get_logger().info(f"当前飞行模式: {self.nav_state}")

    def vehicle_attitude_callback(self, msg):
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def cmdloop_callback(self):
        # 1. 狂发 Offboard 续命信号喵！
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)
        # --- 喵！这里是小菲植入的航点逻辑大脑 ---
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and not self.is_mission_finished:
            self.get_logger().info("喵！飞行中，正在前往航点...")
            if self.current_wp_index < len(self.waypoints):
                target = self.waypoints[self.current_wp_index]
                self.setpoint_position[0] = target[0]
                self.setpoint_position[1] = target[1]
                self.setpoint_position[2] = target[2]
                
                # 计算距离喵！
                dx = self.vehicle_local_position[0] - target[0]
                dy = self.vehicle_local_position[1] - target[1]
                dz = self.vehicle_local_position[2] - target[2]
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                self.get_logger().info(f"喵！当前航点 {self.current_wp_index}: {target}, 距离: {distance}")
                
                # 到达判定喵！
                if distance < self.acceptance_radius:
                    self.get_logger().info(f"喵！成功到达航点 {self.current_wp_index}: {target}")
                    self.current_wp_index += 1
            else:
                self.get_logger().info("🎉 这一圈跑完啦！开启下一圈巡航喵！")
                self.current_wp_index = 1 # 魔法重置！
            # else:
            #     self.get_logger().info("所有航点完成喵！小菲要吃草莓蛋糕！")
            #     self.is_mission_finished = True
        # ------------------------------------------

        # 2. 算误差，丢给 MPC 喵！
        error_position = self.vehicle_local_position - self.setpoint_position

        x0 = np.array([error_position[0], error_position[1], error_position[2],
         self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
         self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10, 1)

        u_pred, x_pred = self.mpc.solve(x0)

        # 3. 把预测轨迹画到 RViz 上喵！
        # idx = 0
        # predicted_path_msg = Path()
        # predicted_path_msg.header.frame_id = 'map'
        # for predicted_state in x_pred:
        #     idx = idx + 1
        #     predicted_pose_msg = PoseStamped()
        #     predicted_pose_msg.pose.position.x = float(predicted_state[0] + self.setpoint_position[0])
        #     predicted_pose_msg.pose.position.y = float(predicted_state[1] + self.setpoint_position[1])
        #     predicted_pose_msg.pose.position.z = float(predicted_state[2] + self.setpoint_position[2])
        #     predicted_path_msg.poses.append(predicted_pose_msg)
        # self.predicted_path_pub.publish(predicted_path_msg)

        # 4. 把底层控制指令发给电机喵！
        thrust_rates = u_pred[0, :]
        thrust_command = -(thrust_rates[0] * 0.07 + 0.0)
        setpoint_msg = VehicleRatesSetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        setpoint_msg.roll = float(thrust_rates[1])
        setpoint_msg.pitch = float(-thrust_rates[2])
        setpoint_msg.yaw = float(-thrust_rates[3])
        setpoint_msg.thrust_body[0] = 0.0
        setpoint_msg.thrust_body[1] = 0.0
        setpoint_msg.thrust_body[2] = float(thrust_command)
        self.publisher_rates_setpoint.publish(setpoint_msg)

def main(args=None):
    rclpy.init(args=args)
    quadrotor_mpc = QuadrotorMPC()
    rclpy.spin(quadrotor_mpc)
    quadrotor_mpc.destroy_node()
    rclpy.shutdown()