#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from mpc_msgs.msg import MissionCommand, PerceptionEvent, PipelineDetection
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, VehicleStatus

from px4_mpc.mission_states import APPROACH, CRUISE, IDLE, INSPECT


DEFAULT_WAYPOINTS = [
    0.0, 0.0, 1.0, 1.57,
    -1.0, -1.0, 1.0, 1.57,
    5.0, -1.0, 1.0, 1.57,
    5.0, 3.5, 1.0, 0.0,
    0.5, 3.5, 1.0, -1.57,
    0.5, 2.0, 1.0, -3.14,
    2.0, 2.0, 1.0, 1.57,
    2.0, 0.5, 1.0, -3.14,
    -1.0, 0.5, 1.0, -1.57,
]


class DecisionBrainNode(Node):

    def __init__(self):
        super().__init__('decision_brain_node')

        self.declare_parameter('mission_command_topic', '/brain/mission_command')
        self.declare_parameter('detection_topic', '/perception/pipeline_detection')
        self.declare_parameter('perception_event_topic', '/perception/event')
        self.declare_parameter('acceptance_radius', 0.5)
        self.declare_parameter('look_ahead_dist', 0.8)
        self.declare_parameter('cruise_speed', 0.3)
        self.declare_parameter('inspection_hold_time_sec', 3.0)
        self.declare_parameter('detection_cooldown_sec', 5.0)
        self.declare_parameter('detection_confidence_threshold', 0.6)
        self.declare_parameter('gas_detection_threshold', 0.2)
        self.declare_parameter('gas_approach_speed', 0.1)
        self.declare_parameter('gas_approach_radius', 0.6)
        self.declare_parameter('gas_keep_current_altitude', True)
        self.declare_parameter('gas_rearm_distance', 2.0)
        self.declare_parameter('loop_waypoints', True)
        self.declare_parameter('waypoints', DEFAULT_WAYPOINTS)

        self.mission_command_topic = self.get_parameter('mission_command_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.perception_event_topic = self.get_parameter('perception_event_topic').value
        self.acceptance_radius = float(self.get_parameter('acceptance_radius').value)
        self.look_ahead_dist = float(self.get_parameter('look_ahead_dist').value)
        self.cruise_speed = float(self.get_parameter('cruise_speed').value)
        self.inspection_hold_time_sec = float(self.get_parameter('inspection_hold_time_sec').value)
        self.detection_cooldown_sec = float(self.get_parameter('detection_cooldown_sec').value)
        self.detection_confidence_threshold = float(self.get_parameter('detection_confidence_threshold').value)
        self.gas_detection_threshold = float(self.get_parameter('gas_detection_threshold').value)
        self.gas_approach_speed = float(self.get_parameter('gas_approach_speed').value)
        self.gas_approach_radius = float(self.get_parameter('gas_approach_radius').value)
        self.gas_keep_current_altitude = bool(self.get_parameter('gas_keep_current_altitude').value)
        self.gas_rearm_distance = float(self.get_parameter('gas_rearm_distance').value)
        self.loop_waypoints = bool(self.get_parameter('loop_waypoints').value)
        self.waypoints = self._load_waypoints(self.get_parameter('waypoints').value)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MANUAL
        self.current_wp_index = 0
        self.mission_finished = False
        self.mission_state = CRUISE

        self.vehicle_local_position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.current_yaw = 0.0

        first_wp = self.waypoints[0]
        self.target_position = np.array(first_wp[:3], dtype=float)
        self.target_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.target_yaw = float(first_wp[3])

        self.inspect_hold_position = None
        self.inspect_hold_yaw = 0.0
        self.inspect_until_sec = None
        self.last_detection_sec = None
        self.approach_target_position = None
        self.approach_target_yaw = 0.0
        self.locked_gas_target_position = None

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile_sub)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.vehicle_attitude_callback, qos_profile_sub)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile_sub)
        self.create_subscription(PipelineDetection, self.detection_topic, self.detection_callback, 10)
        self.create_subscription(PerceptionEvent, self.perception_event_topic, self.perception_event_callback, 10)

        self.command_pub = self.create_publisher(MissionCommand, self.mission_command_topic, 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def _load_waypoints(self, flat_values):
        values = [float(v) for v in flat_values]
        if len(values) < 4 or len(values) % 4 != 0:
            raise ValueError('waypoints parameter must contain x,y,z,yaw groups')
        return [values[i:i + 4] for i in range(0, len(values), 4)]

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

        qw, qx, qy, qz = self.vehicle_attitude
        self.current_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def detection_callback(self, msg):
        if not msg.crack_detected or msg.confidence < self.detection_confidence_threshold:
            return

        self.handle_inspection_trigger(msg.confidence, self.inspection_hold_time_sec)

    def perception_event_callback(self, msg):
        if not msg.detected:
            return
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.get_logger().warn('Perception event received outside OFFBOARD, keep observing and skip state switch.')
            return

        if msg.modality == 'vision':
            if msg.confidence < self.detection_confidence_threshold:
                return
            hold_time = msg.hold_time_sec if msg.hold_time_sec > 0.0 else self.inspection_hold_time_sec
            self.handle_inspection_trigger(msg.confidence, hold_time)
            return

        if msg.modality == 'gas_laser':
            if msg.scalar_value < self.gas_detection_threshold:
                return
            self.handle_gas_event(msg)

    def handle_inspection_trigger(self, confidence, hold_time_sec):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_detection_sec is not None and now_sec - self.last_detection_sec < self.detection_cooldown_sec:
            return

        self.last_detection_sec = now_sec

        self.inspect_hold_position = self.vehicle_local_position.copy()
        self.inspect_hold_yaw = self.current_yaw
        self.inspect_until_sec = now_sec + hold_time_sec
        self.mission_state = INSPECT
        self.get_logger().info(
            f'Switch to INSPECT, confidence={confidence:.2f}, hold for {hold_time_sec:.1f}s.'
        )

    def handle_gas_event(self, msg):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.last_detection_sec is not None and now_sec - self.last_detection_sec < self.detection_cooldown_sec:
            return
        if not msg.position_valid:
            return

        candidate_target = np.array([
            msg.target_position.x,
            msg.target_position.y,
            msg.target_position.z,
        ], dtype=float)
        if self.locked_gas_target_position is not None:
            target_delta = candidate_target - self.locked_gas_target_position
            target_delta[2] = 0.0
            if np.linalg.norm(target_delta) <= self.gas_approach_radius:
                return

        self.last_detection_sec = now_sec
        self.approach_target_position = candidate_target
        if self.gas_keep_current_altitude:
            self.approach_target_position[2] = self.vehicle_local_position[2]
        direction = self.approach_target_position - self.vehicle_local_position
        if np.linalg.norm(direction[:2]) > 1e-6:
            self.approach_target_yaw = math.atan2(direction[1], direction[0])
        else:
            self.approach_target_yaw = self.current_yaw
        self.target_velocity = np.zeros(3)
        self.mission_state = APPROACH
        self.inspect_until_sec = None
        self.get_logger().info(
            f'Switch to APPROACH, gas concentration={msg.scalar_value:.3f}, target leak point received.'
        )

    def timer_callback(self):
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9

        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_command(now, IDLE, self.vehicle_local_position, np.zeros(3), self.current_yaw)
            return

        self.update_gas_rearm_state()

        if self.mission_state == APPROACH:
            self.update_approach_state()
        elif self.mission_state == INSPECT:
            self.update_inspection_state(now_sec)
        else:
            self.update_cruise_state()

        self.publish_command(now, self.mission_state, self.target_position, self.target_velocity, self.target_yaw)

    def update_inspection_state(self, now_sec):
        if self.inspect_until_sec is not None and now_sec >= self.inspect_until_sec:
            self.mission_state = CRUISE
            self.inspect_until_sec = None
            self.inspect_hold_position = None
            if self.approach_target_position is not None:
                self.locked_gas_target_position = self.approach_target_position.copy()
            self.approach_target_position = None
            self.get_logger().info('Inspection window finished, resume cruise.')
            self.update_cruise_state()
            return

        if self.inspect_hold_position is None:
            self.inspect_hold_position = self.vehicle_local_position.copy()
            self.inspect_hold_yaw = self.current_yaw

        self.target_position = self.inspect_hold_position.copy()
        self.target_velocity = np.zeros(3)
        self.target_yaw = self.inspect_hold_yaw

    def update_approach_state(self):
        if self.approach_target_position is None:
            self.mission_state = CRUISE
            self.update_cruise_state()
            return

        delta = self.approach_target_position - self.vehicle_local_position
        horizontal_delta = delta.copy()
        horizontal_delta[2] = 0.0
        horizontal_distance = np.linalg.norm(horizontal_delta)
        if horizontal_distance <= self.gas_approach_radius:
            self.inspect_hold_position = self.vehicle_local_position.copy()
            self.inspect_hold_yaw = self.approach_target_yaw
            self.inspect_until_sec = self.get_clock().now().nanoseconds * 1e-9 + self.inspection_hold_time_sec
            self.mission_state = INSPECT
            self.get_logger().info('Reached gas leak neighborhood, switch to INSPECT hold.')
            self.update_inspection_state(self.get_clock().now().nanoseconds * 1e-9)
            return

        direction = horizontal_delta / max(horizontal_distance, 1e-6)
        speed = min(self.gas_approach_speed, horizontal_distance)
        self.target_position = self.approach_target_position.copy()
        self.target_position[2] = self.vehicle_local_position[2]
        self.target_velocity = direction * speed
        self.target_velocity[2] = 0.0
        self.target_yaw = self.approach_target_yaw

    def update_gas_rearm_state(self):
        if self.locked_gas_target_position is None:
            return

        delta = self.vehicle_local_position - self.locked_gas_target_position
        delta[2] = 0.0
        if np.linalg.norm(delta) >= self.gas_rearm_distance:
            self.get_logger().info('Left inspected gas leak region, gas trigger re-armed.')
            self.locked_gas_target_position = None

    def update_cruise_state(self):
        self.mission_state = CRUISE
        if self.current_wp_index >= len(self.waypoints):
            if self.loop_waypoints and len(self.waypoints) > 1:
                self.current_wp_index = 1
            else:
                self.mission_finished = True
                self.target_position = self.vehicle_local_position.copy()
                self.target_velocity = np.zeros(3)
                self.target_yaw = self.current_yaw
                return

        p_prev = np.array(self.waypoints[self.current_wp_index - 1][:3]) if self.current_wp_index > 0 else self.vehicle_local_position
        p_target = np.array(self.waypoints[self.current_wp_index][:3])

        segment_vec = p_target - p_prev
        seg_len = np.linalg.norm(segment_vec)
        if seg_len > 0.1:
            seg_dir = segment_vec / seg_len
            drone_vec = self.vehicle_local_position - p_prev
            progress = np.dot(drone_vec, seg_dir)
            carrot_progress = min(progress + self.look_ahead_dist, seg_len)
            carrot_point = p_prev + seg_dir * carrot_progress

            self.target_position = carrot_point
            self.target_velocity = seg_dir * self.cruise_speed
            self.target_yaw = math.atan2(seg_dir[1], seg_dir[0])
        else:
            self.target_position = p_target
            self.target_velocity = np.zeros(3)
            self.target_yaw = float(self.waypoints[self.current_wp_index][3])

        if np.linalg.norm(self.vehicle_local_position - p_target) < self.acceptance_radius:
            self.current_wp_index += 1
            self.get_logger().info(f'Advance to waypoint {self.current_wp_index}.')

    def publish_command(self, now, mission_state, target_position, target_velocity, target_yaw):
        msg = MissionCommand()
        msg.stamp = now.to_msg()
        msg.mission_state = mission_state
        msg.active_waypoint_index = self.current_wp_index
        msg.mission_finished = self.mission_finished
        msg.target_position.x = float(target_position[0])
        msg.target_position.y = float(target_position[1])
        msg.target_position.z = float(target_position[2])
        msg.target_velocity.x = float(target_velocity[0])
        msg.target_velocity.y = float(target_velocity[1])
        msg.target_velocity.z = float(target_velocity[2])
        msg.target_yaw = float(target_yaw)
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
