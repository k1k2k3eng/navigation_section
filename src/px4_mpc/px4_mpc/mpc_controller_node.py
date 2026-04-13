#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from mpc_msgs.msg import MissionCommand
from px4_msgs.msg import OffboardControlMode, VehicleAttitude, VehicleLocalPosition, VehicleRatesSetpoint, VehicleStatus

from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC
from px4_mpc.models.multirotor_rate_model import MultirotorRateModel


class ModularMPCControllerNode(Node):

    def __init__(self):
        super().__init__('modular_mpc_controller_node')

        self.declare_parameter('mission_command_topic', '/brain/mission_command')
        self.mission_command_topic = self.get_parameter('mission_command_topic').value

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

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile_sub)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile_sub)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile_sub)
        self.create_subscription(MissionCommand, self.mission_command_topic, self.mission_command_callback, 10)

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile_pub)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL
        self.model = MultirotorRateModel()
        self.mpc = MultirotorRateMPC(self.model)

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.setpoint_position = self.vehicle_local_position.copy()
        self.target_velocity = np.zeros(3, dtype=float)
        self.target_yaw = 0.0
        self.smoothed_target_yaw = 0.0
        self.yaw_initialized = False
        self.command_received = False
        self.latest_state = 'IDLE'

        self.timer = self.create_timer(0.02, self.cmdloop_callback)

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state

    def vehicle_attitude_callback(self, msg):
        q_enu = 1 / np.sqrt(2) * np.array([
            msg.q[0] + msg.q[3],
            msg.q[1] + msg.q[2],
            msg.q[1] - msg.q[2],
            msg.q[0] - msg.q[3],
        ])
        q_enu /= np.linalg.norm(q_enu)
        self.vehicle_attitude = q_enu.astype(float)

        if not self.yaw_initialized:
            self.target_yaw = self.current_yaw()
            self.smoothed_target_yaw = self.target_yaw
            self.yaw_initialized = True

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def mission_command_callback(self, msg):
        self.command_received = True
        self.latest_state = msg.mission_state
        self.setpoint_position = np.array([
            msg.target_position.x,
            msg.target_position.y,
            msg.target_position.z,
        ], dtype=float)
        self.target_velocity = np.array([
            msg.target_velocity.x,
            msg.target_velocity.y,
            msg.target_velocity.z,
        ], dtype=float)
        self.target_yaw = float(msg.target_yaw)

    def current_yaw(self):
        qw, qx, qy, qz = self.vehicle_attitude
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def cmdloop_callback(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)

        if not self.command_received:
            self.setpoint_position = self.vehicle_local_position.copy()
            self.target_velocity = np.zeros(3)
            self.target_yaw = self.current_yaw()

        alpha_yaw = 0.1
        yaw_err = self.target_yaw - self.smoothed_target_yaw
        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
        self.smoothed_target_yaw += alpha_yaw * yaw_err

        current_yaw = self.current_yaw()
        yaw_error = self.smoothed_target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi

        yaw_rate_cmd = np.clip(2.0 * yaw_error, -1.0, 1.0)

        error_position = self.vehicle_local_position - self.setpoint_position
        error_velocity = self.vehicle_local_velocity - self.target_velocity
        x0 = np.array([
            error_position[0], error_position[1], error_position[2],
            error_velocity[0], error_velocity[1], error_velocity[2],
            self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3],
        ]).reshape(10, 1)

        u_pred, _ = self.mpc.solve(x0)
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
    node = ModularMPCControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
