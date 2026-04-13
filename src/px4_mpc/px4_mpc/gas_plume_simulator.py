#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point
from mpc_msgs.msg import GasConcentration
from px4_msgs.msg import VehicleLocalPosition


class GasPlumeSimulatorNode(Node):

    def __init__(self):
        super().__init__('gas_plume_simulator_node')

        self.declare_parameter('gas_concentration_topic', '/perception/gas_concentration')
        self.declare_parameter('leak_position', [2.0, 1.0, 0.0])
        self.declare_parameter('source_strength', 1.0)
        self.declare_parameter('sigma_xy', 1.0)
        self.declare_parameter('sigma_z', 0.5)
        self.declare_parameter('wind_direction_rad', 0.0)
        self.declare_parameter('wind_speed', 1.0)
        self.declare_parameter('measurement_mode', 'tdlas')
        self.declare_parameter('ground_height', 0.0)
        self.declare_parameter('beam_samples', 40)
        self.declare_parameter('absorption_gain', 1.0)
        self.declare_parameter('max_range', 30.0)

        self.gas_concentration_topic = self.get_parameter('gas_concentration_topic').value
        self.leak_position = np.array(self.get_parameter('leak_position').value, dtype=float)
        self.source_strength = float(self.get_parameter('source_strength').value)
        self.sigma_xy = max(float(self.get_parameter('sigma_xy').value), 1e-3)
        self.sigma_z = max(float(self.get_parameter('sigma_z').value), 1e-3)
        self.wind_direction_rad = float(self.get_parameter('wind_direction_rad').value)
        self.wind_speed = max(float(self.get_parameter('wind_speed').value), 1e-3)
        self.measurement_mode = self.get_parameter('measurement_mode').value
        self.ground_height = float(self.get_parameter('ground_height').value)
        self.beam_samples = max(int(self.get_parameter('beam_samples').value), 2)
        self.absorption_gain = max(float(self.get_parameter('absorption_gain').value), 1e-6)
        self.max_range = max(float(self.get_parameter('max_range').value), 1e-3)

        self.vehicle_position = np.array([0.0, 0.0, 0.0], dtype=float)

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )
        self.concentration_pub = self.create_publisher(GasConcentration, self.gas_concentration_topic, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, msg):
        self.vehicle_position[0] = msg.y
        self.vehicle_position[1] = msg.x
        self.vehicle_position[2] = -msg.z

    def timer_callback(self):
        concentration = self.compute_concentration(self.vehicle_position)

        msg = GasConcentration()
        msg.stamp = self.get_clock().now().to_msg()
        msg.source_frame_id = 'map'
        msg.vehicle_position = self.vector_to_point(self.vehicle_position)
        msg.leak_position = self.vector_to_point(self.leak_position)
        msg.concentration = float(concentration)
        self.concentration_pub.publish(msg)

    def compute_concentration(self, vehicle_position):
        if self.measurement_mode == 'point':
            return max(self.gas_density_at_point(vehicle_position), 0.0)

        if self.measurement_mode == 'column':
            # Legacy approximation: integrate vertically without explicit absorption mapping.
            path_integral = self.integrate_vertical_column(vehicle_position)
            return max(path_integral, 0.0)

        # TDLAS-style proxy:
        # 1. Integrate plume density along the laser path.
        # 2. Convert the path integral to an absorbance-like quantity.
        path_integral = self.integrate_vertical_column(vehicle_position)
        absorbance_proxy = 1.0 - math.exp(-self.absorption_gain * path_integral)
        return max(absorbance_proxy, 0.0)

    def integrate_vertical_column(self, vehicle_position):
        start = vehicle_position.copy()
        end = vehicle_position.copy()
        end[2] = self.ground_height

        path = end - start
        path_length = min(np.linalg.norm(path), self.max_range)
        if path_length <= 1e-6:
            return 0.0

        direction = path / np.linalg.norm(path)
        step = path_length / float(self.beam_samples)
        integral = 0.0

        for idx in range(self.beam_samples + 1):
            distance = min(idx * step, path_length)
            sample_point = start + direction * distance
            weight = 0.5 if idx in (0, self.beam_samples) else 1.0
            integral += weight * self.gas_density_at_point(sample_point)

        return integral * step

    def gas_density_at_point(self, point):
        rel = point - self.leak_position
        wind_dir = np.array([math.cos(self.wind_direction_rad), math.sin(self.wind_direction_rad), 0.0], dtype=float)
        downwind = float(np.dot(rel, wind_dir))
        crosswind_vec = rel - downwind * wind_dir
        crosswind = np.linalg.norm(crosswind_vec[:2])
        dz = rel[2]

        attenuation = math.exp(-0.5 * ((crosswind / self.sigma_xy) ** 2 + (dz / self.sigma_z) ** 2))
        downwind_gain = math.exp(-max(-downwind, 0.0) / self.sigma_xy) / (1.0 + max(downwind, 0.0) / self.wind_speed)
        return self.source_strength * attenuation * downwind_gain

    def vector_to_point(self, vector):
        point = Point()
        point.x = float(vector[0])
        point.y = float(vector[1])
        point.z = float(vector[2])
        return point


def main(args=None):
    rclpy.init(args=args)
    node = GasPlumeSimulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
