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
        super().__init__('quadrotor_mpc_node')

        self.waypoints = [[0.0, 0.0, 1.0, 1.57],[-1.0, -1.0, 1.0, 1.57],[5.0, -1.0, 1.0, 1.57],[5.0, 3.5, 1.0, 0.0],[0.5, 3.5, 1.0, -1.57],[0.5, 2.0, 1.0, -3.14],[2.0, 2.0, 1.0, 1.57],[2.0, 0.5, 1.0, -3.14],[-1.0, 0.5, 1.0, -1.57]]
        self.current_wp_index = 0
        self.acceptance_radius = 0.5 # 👑 略微增大，配合前瞻点实现平滑切弯
        self.is_mission_finished = False
        
        # 👑 丝滑导航核心参数
        self.look_ahead_dist = 0.6 # 前瞻距离（胡萝卜距离），越大越平滑，越小越贴合路径
        self.target_yaw = 0.0 
        self.smoothed_target_yaw = 0.0
        self.yaw_initialized = False

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

        # Timer (50Hz)
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL

        # Create MPC
        self.model = MultirotorRateModel()
        self.mpc = MultirotorRateMPC(self.model)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        
        self.setpoint_position = np.array([self.waypoints[0][0], self.waypoints[0][1], self.waypoints[0][2]])

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def vehicle_attitude_callback(self, msg):
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)
        
        # 👑 魔法：在起飞瞬间记录真实航向，防止起跳时猛回头喵
        if not self.yaw_initialized:
            qw, qx, qy, qz = self.vehicle_attitude
            current_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            self.target_yaw = current_yaw
            self.smoothed_target_yaw = current_yaw
            self.yaw_initialized = True

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def cmdloop_callback(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and not self.is_mission_finished:
            if self.current_wp_index < len(self.waypoints):
                # 1. 确定当前段的起点和终点
                p_prev = np.array(self.waypoints[self.current_wp_index-1][:3]) if self.current_wp_index > 0 else self.vehicle_local_position
                p_target = np.array(self.waypoints[self.current_wp_index][:3])
                
                # 2. 计算航段向量和胡萝卜点 (Carrot Point)
                segment_vec = p_target - p_prev
                seg_len = np.linalg.norm(segment_vec)
                
                if seg_len > 0.1:
                    seg_dir = segment_vec / seg_len
                    # 无人机在航段上的投影进度
                    drone_vec = self.vehicle_local_position - p_prev
                    progress = np.dot(drone_vec, seg_dir)
                    
                    # 沿着航段向前看一段距离作为胡萝卜点
                    carrot_progress = min(progress + self.look_ahead_dist, seg_len)
                    carrot_point = p_prev + seg_dir * carrot_progress
                    
                    # 设定 MPC 追踪这个滑动的点，而不是死航点喵！
                    self.setpoint_position = carrot_point
                    
                    # 锁定航向为航段方向
                    self.target_yaw = math.atan2(seg_dir[1], seg_dir[0])
                else:
                    self.setpoint_position = p_target

                # 3. 检查是否切换航点（离真正航点足够近了）
                dist_to_wp = np.linalg.norm(self.vehicle_local_position - p_target)
                if dist_to_wp < self.acceptance_radius:
                    self.current_wp_index += 1
                    print(f"Switching to waypoint {self.current_wp_index}...")
            else:
                self.current_wp_index = 1

        # 👑 魔法2：平滑偏航角指令（Low-pass Filter）
        alpha_yaw = 0.1 # 越小越丝滑，越大转向越灵敏喵
        yaw_err = self.target_yaw - self.smoothed_target_yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
        self.smoothed_target_yaw += alpha_yaw * yaw_err

        qw, qx, qy, qz = self.vehicle_attitude
        current_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # 计算角度反馈误差
        yaw_error = self.smoothed_target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
        
        # P控制器增益
        k_yaw = 2.0 
        yaw_rate_cmd = k_yaw * yaw_error
        
        # 限制最大转向速度
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0)


        # 还原最原始无暇的 MPC 状态输入，让它专心飞位置喵！
        error_position = self.vehicle_local_position - self.setpoint_position
        x0 = np.array([error_position[0], error_position[1], error_position[2],
         self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
         self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10, 1)

        u_pred, x_pred = self.mpc.solve(x0)

        thrust_rates = u_pred[0, :]
        thrust_command = -(thrust_rates[0] * 0.07 + 0.0)
        
        setpoint_msg = VehicleRatesSetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        # 听 MPC 的横滚和俯仰（负责飞往目标）
        setpoint_msg.roll = float(thrust_rates[1])
        setpoint_msg.pitch = float(-thrust_rates[2])
        
        # 👑 魔法3：强行覆盖 MPC 的偏航角速度，用小菲自己算的！
        # (因为底层是 FRD 下为正，ENU 上为正，所以加个负号)
        setpoint_msg.yaw = float(-yaw_rate_cmd) 
        
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

if __name__ == '__main__':
    main()
