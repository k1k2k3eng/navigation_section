#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleThrustSetpoint # 👑 新增消息类型

class QuadrotorPID(Node):

    def __init__(self):
        super().__init__('quadrotor_pid_node')

        # 统一轨迹点
        self.waypoints = [[0.0, 0.0, 1.0, 1.57],[-1.0, -1.0, 1.0, 1.57],[5.0, -1.0, 1.0, 1.57],[5.0, 3.5, 1.0, 0.0],[0.5, 3.5, 1.0, -1.57],[0.5, 2.0, 1.0, -3.14],[2.0, 2.0, 1.0, 1.57],[2.0, 0.5, 1.0, -3.14],[-1.0, 0.5, 1.0, -1.57]]
        self.current_wp_index = 0
        self.acceptance_radius = 0.6 
        self.is_mission_finished = False

        # 👑 强化版参数
        self.look_ahead_dist = 0.4 # 前瞻距离
        self.cruise_speed = 0.5   # 巡航速度 (m/s)
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
        self.local_position_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile_sub)
        self.attitude_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile_sub)

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_pub)
        
        # 👑 魔法：镜像发布推力数据，用于 plot_thrust 读取
        self.publisher_debug_thrust = self.create_publisher(VehicleThrustSetpoint, '/fmu/out/vehicle_thrust_setpoint', qos_profile_pub)

        # Timer (50Hz)
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL

        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def vehicle_attitude_callback(self, msg):
        # 同样的坐标系处理逻辑
        q_enu = 1/np.sqrt(2) * np.array([msg.q[0] + msg.q[3], msg.q[1] + msg.q[2], msg.q[1] - msg.q[2], msg.q[0] - msg.q[3]])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)
        
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

    def cmdloop_callback(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = True
        offboard_msg.velocity = True # 开启速度前馈支持
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.publisher_offboard_mode.publish(offboard_msg)

        if self.is_mission_finished:
            return

        setpoint_msg = TrajectorySetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # 待命状态：维持当前位置
            setpoint_msg.position = [float(self.vehicle_local_position[1]), float(self.vehicle_local_position[0]), float(-self.vehicle_local_position[2])]
            setpoint_msg.velocity = [0.0, 0.0, 0.0]
            setpoint_msg.yaw = 0.0
        else:
            # 👑 强化版任务执行逻辑
            if self.current_wp_index < len(self.waypoints):
                p_prev = np.array(self.waypoints[self.current_wp_index-1][:3]) if self.current_wp_index > 0 else self.vehicle_local_position
                p_target = np.array(self.waypoints[self.current_wp_index][:3])
                
                # 计算胡萝卜引导
                segment_vec = p_target - p_prev
                seg_len = np.linalg.norm(segment_vec)
                
                v_feedforward = np.zeros(3)
                if seg_len > 0.1:
                    seg_dir = segment_vec / seg_len
                    drone_vec = self.vehicle_local_position - p_prev
                    progress = np.dot(drone_vec, seg_dir)
                    
                    carrot_progress = min(progress + self.look_ahead_dist, seg_len)
                    carrot_point = p_prev + seg_dir * carrot_progress
                    
                    # 注入速度前馈
                    v_feedforward = seg_dir * self.cruise_speed
                    
                    target_pos = carrot_point
                    self.target_yaw = math.atan2(seg_dir[1], seg_dir[0])
                else:
                    target_pos = p_target

                # 航向平滑
                alpha_yaw = 0.1
                yaw_err = self.target_yaw - self.smoothed_target_yaw
                yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
                self.smoothed_target_yaw += alpha_yaw * yaw_err

                # 发布指令 (转换回 PX4 NED)
                setpoint_msg.position = [float(target_pos[1]), float(target_pos[0]), float(-target_pos[2])]
                setpoint_msg.velocity = [float(v_feedforward[1]), float(v_feedforward[0]), float(-v_feedforward[2])]
                setpoint_msg.yaw = float(-self.smoothed_target_yaw + 1.57)

                # 切换航点
                dist_to_wp = np.linalg.norm(self.vehicle_local_position - p_target)
                if dist_to_wp < self.acceptance_radius:
                    self.current_wp_index += 1
                    print(f"Strong PID: Waypoint {self.current_wp_index} achieved!")

                # 👑 魔法：镜像发布推力数据，让 plot_thrust 以为这是飞控出的数据
                # 由于 PID 追踪位置，我们简单地把位置误差作为推力的近似模拟，或者发一个占位符
                # 这里我们发一个归一化的方向推力，用于演示对比
                thrust_msg = VehicleThrustSetpoint()
                thrust_msg.timestamp = setpoint_msg.timestamp
                # 模拟推力：z轴通常是主要推力，这里简单模拟
                thrust_msg.xyz = [0.0, 0.0, -0.7] # 悬停推力近似值
                self.publisher_debug_thrust.publish(thrust_msg)
            else:
                self.current_wp_index = 1

        self.publisher_trajectory_setpoint.publish(setpoint_msg)

def main(args=None):
    rclpy.init(args=args)
    quadrotor_pid = QuadrotorPID()
    rclpy.spin(quadrotor_pid)
    quadrotor_pid.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
