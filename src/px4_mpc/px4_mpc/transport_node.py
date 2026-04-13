#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from mpc_msgs.msg import MissionCommand, PerceptionEvent, PipelineDetection

from px4_mpc.mission_states import ARCHIVE_TRIGGER_STATES


class TransportNode(Node):

    def __init__(self):
        super().__init__('mission_transport_node')

        self.declare_parameter('mission_command_topic', '/brain/mission_command')
        self.declare_parameter('detection_topic', '/perception/pipeline_detection')
        self.declare_parameter('perception_event_topic', '/perception/event')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('archive_dir', '/tmp/px4_mpc_archive')
        self.declare_parameter('archive_images', True)
        self.declare_parameter('detection_confidence_threshold', 0.6)
        self.declare_parameter('qgc_status_topic', '/transport/qgc_status_text')

        self.mission_command_topic = self.get_parameter('mission_command_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.perception_event_topic = self.get_parameter('perception_event_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.archive_dir = Path(self.get_parameter('archive_dir').value)
        self.archive_images = bool(self.get_parameter('archive_images').value)
        self.detection_confidence_threshold = float(self.get_parameter('detection_confidence_threshold').value)
        self.qgc_status_topic = self.get_parameter('qgc_status_topic').value

        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.current_state = ''
        self.last_detection = None
        self.last_perception_event = None
        self.latest_image = None
        self.archived_for_current_state = False

        self.create_subscription(MissionCommand, self.mission_command_topic, self.mission_command_callback, 10)
        self.create_subscription(PipelineDetection, self.detection_topic, self.detection_callback, 10)
        self.create_subscription(PerceptionEvent, self.perception_event_topic, self.perception_event_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.qgc_status_pub = self.create_publisher(String, self.qgc_status_topic, 10)

    def mission_command_callback(self, msg):
        if msg.mission_state != self.current_state:
            self.current_state = msg.mission_state
            self.archived_for_current_state = False

        if self.current_state in ARCHIVE_TRIGGER_STATES:
            self.maybe_archive(msg)

    def detection_callback(self, msg):
        self.last_detection = msg

    def perception_event_callback(self, msg):
        self.last_perception_event = msg

    def image_callback(self, msg):
        self.latest_image = msg

    def maybe_archive(self, command_msg):
        if self.archived_for_current_state:
            return
        if self.last_perception_event is not None and self.last_perception_event.detected:
            confidence = self.last_perception_event.confidence
        elif self.last_detection is not None and self.last_detection.crack_detected:
            confidence = self.last_detection.confidence
        else:
            return
        if confidence < self.detection_confidence_threshold:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        stem = self.archive_dir / f'inspection_{timestamp}'
        metadata = {
            'mission_state': command_msg.mission_state,
            'active_waypoint_index': int(command_msg.active_waypoint_index),
            'target_position': {
                'x': float(command_msg.target_position.x),
                'y': float(command_msg.target_position.y),
                'z': float(command_msg.target_position.z),
            },
        }
        if self.last_perception_event is not None and self.last_perception_event.detected:
            metadata['perception_event'] = {
                'modality': self.last_perception_event.modality,
                'event_type': self.last_perception_event.event_type,
                'confidence': float(self.last_perception_event.confidence),
                'scalar_value': float(self.last_perception_event.scalar_value),
                'position_valid': bool(self.last_perception_event.position_valid),
                'target_position': {
                    'x': float(self.last_perception_event.target_position.x),
                    'y': float(self.last_perception_event.target_position.y),
                    'z': float(self.last_perception_event.target_position.z),
                },
            }
        elif self.last_detection is not None:
            metadata['detection'] = {
                'confidence': float(self.last_detection.confidence),
                'class_name': self.last_detection.class_name,
                'center_x': float(self.last_detection.center_x),
                'center_y': float(self.last_detection.center_y),
                'bbox_xmin': float(self.last_detection.bbox_xmin),
                'bbox_ymin': float(self.last_detection.bbox_ymin),
                'bbox_xmax': float(self.last_detection.bbox_xmax),
                'bbox_ymax': float(self.last_detection.bbox_ymax),
            }

        metadata_path = stem.with_suffix('.json')
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2))

        image_path = None
        if self.archive_images and self.latest_image is not None:
            image_path = self.save_image(self.latest_image, stem)

        event_text = f'Inspection archived at {metadata_path}'
        if image_path is not None:
            event_text += f', image={image_path}'
        self.publish_qgc_status(event_text)
        self.get_logger().info(event_text)
        self.archived_for_current_state = True

    def publish_qgc_status(self, text):
        msg = String()
        msg.data = text
        self.qgc_status_pub.publish(msg)

    def save_image(self, msg, stem):
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            raw_path = stem.with_suffix('.raw')
            raw_path.write_bytes(bytes(msg.data))
            return str(raw_path)

        image_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == 'mono8':
            row = image_bytes.reshape(msg.height, msg.step)[:, :msg.width]
            image_path = stem.with_suffix('.pgm')
            with image_path.open('wb') as handle:
                handle.write(f'P5\n{msg.width} {msg.height}\n255\n'.encode('ascii'))
                handle.write(row.tobytes())
            return str(image_path)

        channels = 3
        row = image_bytes.reshape(msg.height, msg.step)[:, :msg.width * channels]
        image = row.reshape(msg.height, msg.width, channels)
        if msg.encoding == 'bgr8':
            image = image[:, :, ::-1]

        image_path = stem.with_suffix('.ppm')
        with image_path.open('wb') as handle:
            handle.write(f'P6\n{msg.width} {msg.height}\n255\n'.encode('ascii'))
            handle.write(image.tobytes())
        return str(image_path)


def main(args=None):
    rclpy.init(args=args)
    node = TransportNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
