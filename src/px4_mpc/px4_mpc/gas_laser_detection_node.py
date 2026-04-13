#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from mpc_msgs.msg import GasConcentration, PerceptionEvent


class GasLaserDetectionNode(Node):

    def __init__(self):
        super().__init__('gas_laser_detection_node')

        self.declare_parameter('gas_concentration_topic', '/perception/gas_concentration')
        self.declare_parameter('perception_event_topic', '/perception/event')
        self.declare_parameter('gas_detection_threshold', 0.2)
        self.declare_parameter('gas_event_confidence_scale', 1.5)
        self.declare_parameter('approach_speed', 0.1)
        self.declare_parameter('inspection_hold_time_sec', 3.0)

        self.gas_concentration_topic = self.get_parameter('gas_concentration_topic').value
        self.perception_event_topic = self.get_parameter('perception_event_topic').value
        self.gas_detection_threshold = float(self.get_parameter('gas_detection_threshold').value)
        self.gas_event_confidence_scale = float(self.get_parameter('gas_event_confidence_scale').value)
        self.approach_speed = float(self.get_parameter('approach_speed').value)
        self.inspection_hold_time_sec = float(self.get_parameter('inspection_hold_time_sec').value)

        self.create_subscription(GasConcentration, self.gas_concentration_topic, self.gas_concentration_callback, 10)
        self.perception_event_pub = self.create_publisher(PerceptionEvent, self.perception_event_topic, 10)

    def gas_concentration_callback(self, msg):
        event = PerceptionEvent()
        event.stamp = msg.stamp
        event.source_frame_id = msg.source_frame_id
        event.modality = 'gas_laser'
        event.event_type = 'gas_leak'
        event.scalar_value = float(msg.concentration)
        event.detected = msg.concentration >= self.gas_detection_threshold
        event.confidence = self.concentration_to_confidence(msg.concentration)
        event.position_valid = event.detected
        event.target_position = msg.leak_position
        event.hold_time_sec = self.inspection_hold_time_sec
        event.recommended_speed = self.approach_speed
        self.perception_event_pub.publish(event)

    def concentration_to_confidence(self, concentration):
        if concentration <= 0.0:
            return 0.0
        scaled = concentration * self.gas_event_confidence_scale
        return float(max(0.0, min(1.0, 1.0 - math.exp(-scaled))))


def main(args=None):
    rclpy.init(args=args)
    node = GasLaserDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
