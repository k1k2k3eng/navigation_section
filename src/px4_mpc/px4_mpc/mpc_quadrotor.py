#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from enum import Enum
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from std_msgs.msg import Bool

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint

from mpc_msgs.srv import SetPose

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC


class MissionState(Enum):
    CRUISE = "cruise"
    INSPECT = "inspect"


class QuadrotorMPC(Node):

    def __init__(self):
        super().__init__('quadrotor_mpc_node')

        self.waypoints = [[0.0, 0.0, 1.0, 1.57],[-1.0, -1.0, 1.0, 1.57],[5.0, -1.0, 1.0, 1.57],[5.0, 3.5, 1.0, 0.0],[0.5, 3.5, 1.0, -1.57],[0.5, 2.0, 1.0, -3.14],[2.0, 2.0, 1.0, 1.57],[2.0, 0.5, 1.0, -3.14],[-1.0, 0.5, 1.0, -1.57]]
        self.current_wp_index = 0
        self.acceptance_radius = 0.5 
        self.is_mission_finished = False
        
        # 👑 丝滑导航核心参数
        self.look_ahead_dist = 0.8 
        self.cruise_speed = 0.3 
        self.target_yaw = 0.0 
        self.smoothed_target_yaw = 0.0
        self.yaw_initialized = False
        self.mission_state = MissionState.CRUISE
        self.inspect_hold_position = None
        self.inspect_until_sec = None
        self.last_crack_detection_sec = None

        self.declare_parameter('crack_detection_topic', '/vision/crack_detected')
        self.declare_parameter('photo_trigger_topic', '/inspection/photo_trigger')
        self.declare_parameter('inspection_hold_time_sec', 3.0)
        self.declare_parameter('detection_cooldown_sec', 5.0)
        self.declare_parameter('enable_photo_trigger', True)

        self.crack_detection_topic = self.get_parameter('crack_detection_topic').value
        self.photo_trigger_topic = self.get_parameter('photo_trigger_topic').value
        self.inspection_hold_time_sec = float(self.get_parameter('inspection_hold_time_sec').value)
        self.detection_cooldown_sec = float(self.get_parameter('detection_cooldown_sec').value)
        self.enable_photo_trigger = bool(self.get_parameter('enable_photo_trigger').value)

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
        self.crack_detected_sub = self.create_subscription(Bool, self.crack_detection_topic, self.crack_detected_callback, 10)

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile_pub)
        self.photo_trigger_pub = self.create_publisher(Bool, self.photo_trigger_topic, 10)

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
        self.target_velocity = np.array([0.0, 0.0, 0.0]) 

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def vehicle_attitude_callback(self, msg):
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
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def crack_detected_callback(self, msg):
        if not msg.data:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_crack_detection_sec is not None:
            if now_sec - self.last_crack_detection_sec < self.detection_cooldown_sec:
                return

        self.last_crack_detection_sec = now_sec

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().warn('Crack detected, but vehicle is not in OFFBOARD. Ignoring trigger.')
            return

        if self.mission_state == MissionState.INSPECT:
            self.inspect_until_sec = now_sec + self.inspection_hold_time_sec
            self.get_logger().info('Crack detected again during inspection, extending hold time.')
            return

        self.start_inspection(now_sec)

    def start_inspection(self, now_sec):
        self.mission_state = MissionState.INSPECT
        self.inspect_hold_position = self.vehicle_local_position.copy()
        self.setpoint_position = self.inspect_hold_position.copy()
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.inspect_until_sec = now_sec + self.inspection_hold_time_sec

        qw, qx, qy, qz = self.vehicle_attitude
        self.target_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        if self.enable_photo_trigger:
            trigger_msg = Bool()
            trigger_msg.data = True
            self.photo_trigger_pub.publish(trigger_msg)

        self.get_logger().info(
            f'Crack detected, hold position for {self.inspection_hold_time_sec:.1f}s and trigger capture.'
        )

    def finish_inspection_if_needed(self):
        if self.mission_state != MissionState.INSPECT or self.inspect_until_sec is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec < self.inspect_until_sec:
            self.setpoint_position = self.inspect_hold_position.copy()
            self.target_velocity = np.array([0.0, 0.0, 0.0])
            return

        self.mission_state = MissionState.CRUISE
        self.inspect_hold_position = None
        self.inspect_until_sec = None
        self.get_logger().info('Inspection finished, resume waypoint cruise.')

    def update_cruise_setpoint(self):
        if self.current_wp_index < len(self.waypoints):
            p_prev = np.array(self.waypoints[self.current_wp_index-1][:3]) if self.current_wp_index > 0 else self.vehicle_local_position
            p_target = np.array(self.waypoints[self.current_wp_index][:3])
            
            segment_vec = p_target - p_prev
            seg_len = np.linalg.norm(segment_vec)
            
            if seg_len > 0.1:
                seg_dir = segment_vec / seg_len
                drone_vec = self.vehicle_local_position - p_prev
                progress = np.dot(drone_vec, seg_dir)
                
                carrot_progress = min(progress + self.look_ahead_dist, seg_len)
                carrot_point = p_prev + seg_dir * carrot_progress
                
                self.setpoint_position = carrot_point
                self.target_velocity = seg_dir * self.cruise_speed
                self.target_yaw = math.atan2(seg_dir[1], seg_dir[0])
            else:
                self.setpoint_position = p_target
                self.target_velocity = np.array([0.0, 0.0, 0.0])

            dist_to_wp = np.linalg.norm(self.vehicle_local_position - p_target)
            if dist_to_wp < self.acceptance_radius:
                self.current_wp_index += 1
                self.get_logger().info(f'Switching to waypoint {self.current_wp_index}...')
        else:
            self.current_wp_index = 1

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
            self.finish_inspection_if_needed()
            if self.mission_state == MissionState.CRUISE:
                self.update_cruise_setpoint()

        alpha_yaw = 0.1 
        yaw_err = self.target_yaw - self.smoothed_target_yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
        self.smoothed_target_yaw += alpha_yaw * yaw_err

        qw, qx, qy, qz = self.vehicle_attitude
        current_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        yaw_error = self.smoothed_target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
        
        k_yaw = 2.0 
        yaw_rate_cmd = k_yaw * yaw_error
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0)

        error_position = self.vehicle_local_position - self.setpoint_position
        error_velocity = self.vehicle_local_velocity - self.target_velocity 
        
        x0 = np.array([error_position[0], error_position[1], error_position[2],
         error_velocity[0], error_velocity[1], error_velocity[2],
         self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10, 1)

        u_pred, x_pred = self.mpc.solve(x0)

        thrust_rates = u_pred[0, :]
        thrust_command = -(thrust_rates[0] * 0.07 + 0.0)
        
        setpoint_msg = VehicleRatesSetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        
        setpoint_msg.roll = float(thrust_rates[1])
        setpoint_msg.pitch = float(-thrust_rates[2])
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
