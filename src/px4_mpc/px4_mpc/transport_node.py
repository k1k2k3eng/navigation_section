#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from mpc_msgs.msg import MissionCommand, PerceptionEvent, PipelineDetection
from px4_msgs.msg import VehicleLocalPosition

from px4_mpc.mission_states import ARCHIVE_TRIGGER_STATES


class TransportNode(Node):

    def __init__(self):
        super().__init__('mission_transport_node')

        self.declare_parameter('mission_command_topic', '/brain/mission_command')
        self.declare_parameter('detection_topic', '/perception/pipeline_detection')
        self.declare_parameter('perception_event_topic', '/perception/event')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('archive_dir', '/home/kkk/navigation_section/navigation_ws/src/px4_mpc/px4_mpc/archive')
        self.declare_parameter('archive_images', True)
        self.declare_parameter('detection_confidence_threshold', 0.6)
        self.declare_parameter('gas_scalar_threshold', 0.2)
        self.declare_parameter('qgc_status_topic', '/transport/qgc_status_text')
        self.declare_parameter('capture_delay_sec', 0.8)
        self.declare_parameter('max_hover_speed', 0.15)
        self.declare_parameter('thumbnail_max_dim', 320)

        self.mission_command_topic = self.get_parameter('mission_command_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.perception_event_topic = self.get_parameter('perception_event_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.archive_dir = Path(self.get_parameter('archive_dir').value)
        self.archive_images = bool(self.get_parameter('archive_images').value)
        self.detection_confidence_threshold = float(self.get_parameter('detection_confidence_threshold').value)
        self.gas_scalar_threshold = float(self.get_parameter('gas_scalar_threshold').value)
        self.qgc_status_topic = self.get_parameter('qgc_status_topic').value
        self.capture_delay_sec = float(self.get_parameter('capture_delay_sec').value)
        self.max_hover_speed = float(self.get_parameter('max_hover_speed').value)
        self.thumbnail_max_dim = max(int(self.get_parameter('thumbnail_max_dim').value), 32)

        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.current_state = ''
        self.current_state_entered_sec = None
        self.latest_command = None
        self.last_detection = None
        self.last_perception_event = None
        self.latest_image = None
        self.archived_for_current_state = False
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_wait_reason = None

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(MissionCommand, self.mission_command_topic, self.mission_command_callback, 10)
        self.create_subscription(PipelineDetection, self.detection_topic, self.detection_callback, 10)
        self.create_subscription(PerceptionEvent, self.perception_event_topic, self.perception_event_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile_sub,
        )
        self.qgc_status_pub = self.create_publisher(String, self.qgc_status_topic, 10)
        self.create_timer(0.1, self.archive_timer_callback)

    def mission_command_callback(self, msg):
        self.latest_command = msg
        if msg.mission_state != self.current_state:
            self.current_state = msg.mission_state
            self.current_state_entered_sec = self.now_sec()
            self.archived_for_current_state = False

    def detection_callback(self, msg):
        self.last_detection = msg

    def perception_event_callback(self, msg):
        self.last_perception_event = msg

    def image_callback(self, msg):
        self.latest_image = msg

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def archive_timer_callback(self):
        if self.latest_command is None:
            return
        if self.current_state not in ARCHIVE_TRIGGER_STATES:
            self.last_wait_reason = None
            return
        self.maybe_archive(self.latest_command)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def is_hover_stable(self):
        return float(np.linalg.norm(self.vehicle_local_velocity)) <= self.max_hover_speed

    def maybe_archive(self, command_msg):
        if self.archived_for_current_state:
            return
        if self.current_state_entered_sec is None:
            return
        if self.now_sec() - self.current_state_entered_sec < self.capture_delay_sec:
            self.report_wait_reason('waiting_capture_delay')
            return
        if not self.is_hover_stable():
            self.report_wait_reason('waiting_hover_stable')
            return
        confidence = None
        if self.last_perception_event is not None and self.last_perception_event.detected:
            if self.last_perception_event.modality == 'gas_laser':
                if self.last_perception_event.scalar_value < self.gas_scalar_threshold:
                    self.report_wait_reason('waiting_gas_scalar_threshold')
                    return
                confidence = self.last_perception_event.confidence
            else:
                confidence = self.last_perception_event.confidence
                if confidence < self.detection_confidence_threshold:
                    self.report_wait_reason('waiting_confidence_threshold')
                    return
        elif self.last_detection is not None and self.last_detection.crack_detected:
            confidence = self.last_detection.confidence
            if confidence < self.detection_confidence_threshold:
                self.report_wait_reason('waiting_confidence_threshold')
                return
        else:
            self.report_wait_reason('waiting_detection_event')
            return

        self.last_wait_reason = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        stem = self.archive_dir / f'inspection_{timestamp}'
        metadata = {
            'mission_state': command_msg.mission_state,
            'active_waypoint_index': int(command_msg.active_waypoint_index),
            'captured_at': timestamp,
            'vehicle_position': {
                'x': float(self.vehicle_local_position[0]),
                'y': float(self.vehicle_local_position[1]),
                'z': float(self.vehicle_local_position[2]),
            },
            'vehicle_velocity': {
                'x': float(self.vehicle_local_velocity[0]),
                'y': float(self.vehicle_local_velocity[1]),
                'z': float(self.vehicle_local_velocity[2]),
                'norm': float(np.linalg.norm(self.vehicle_local_velocity)),
            },
            'target_position': {
                'x': float(command_msg.target_position.x),
                'y': float(command_msg.target_position.y),
                'z': float(command_msg.target_position.z),
            },
            'image_available': bool(self.latest_image is not None),
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
        full_image_path = None
        thumbnail_path = None
        if self.archive_images and self.latest_image is not None:
            full_image_path, thumbnail_path = self.save_image_bundle(self.latest_image, stem)

        metadata['artifacts'] = {
            'full_image_path': full_image_path,
            'thumbnail_path': thumbnail_path,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2))

        summary = {
            'event': 'inspection_capture',
            'mission_state': command_msg.mission_state,
            'waypoint_index': int(command_msg.active_waypoint_index),
            'vehicle_position': metadata['vehicle_position'],
            'confidence': float(confidence),
            'image_available': bool(self.latest_image is not None),
            'thumbnail_path': thumbnail_path,
        }
        event_text = json.dumps(summary, ensure_ascii=True)
        self.publish_qgc_status(event_text)
        if self.latest_image is None:
            self.get_logger().warn(f'Inspection archived without image at {metadata_path}')
        else:
            self.get_logger().info(f'Inspection archived at {metadata_path}')
        self.archived_for_current_state = True

    def report_wait_reason(self, reason):
        if self.last_wait_reason == reason:
            return
        self.last_wait_reason = reason
        self.get_logger().info(f'Transport waiting: {reason}')

    def publish_qgc_status(self, text):
        msg = String()
        msg.data = text
        self.qgc_status_pub.publish(msg)

    def save_image_bundle(self, msg, stem):
        image_array, extension = self.image_msg_to_array(msg)
        if image_array is None:
            raw_path = stem.with_suffix('.raw')
            raw_path.write_bytes(bytes(msg.data))
            return str(raw_path), None

        full_path = self.write_array_image(image_array, stem.with_name(f'{stem.name}_full'), extension)
        thumbnail = self.make_thumbnail(image_array)
        thumb_path = self.write_array_image(thumbnail, stem.with_name(f'{stem.name}_thumb'), extension)
        return full_path, thumb_path

    def image_msg_to_array(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            return None, None

        image_bytes = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == 'mono8':
            row = image_bytes.reshape(msg.height, msg.step)[:, :msg.width]
            return row.copy(), '.pgm'

        channels = 3
        row = image_bytes.reshape(msg.height, msg.step)[:, :msg.width * channels]
        image = row.reshape(msg.height, msg.width, channels)
        if msg.encoding == 'bgr8':
            image = image[:, :, ::-1]
        return image.copy(), '.ppm'

    def write_array_image(self, image, stem, extension):
        image_path = stem.with_suffix(extension)
        with image_path.open('wb') as handle:
            if image.ndim == 2:
                height, width = image.shape
                handle.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
                handle.write(image.tobytes())
            else:
                height, width, _ = image.shape
                handle.write(f'P6\n{width} {height}\n255\n'.encode('ascii'))
                handle.write(image.tobytes())
        return str(image_path)

    def make_thumbnail(self, image):
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape

        scale = min(1.0, float(self.thumbnail_max_dim) / max(height, width))
        if scale >= 1.0:
            return image

        new_height = max(1, int(height * scale))
        new_width = max(1, int(width * scale))
        y_idx = np.linspace(0, height - 1, new_height).astype(int)
        x_idx = np.linspace(0, width - 1, new_width).astype(int)
        if image.ndim == 2:
            return image[np.ix_(y_idx, x_idx)]
        return image[np.ix_(y_idx, x_idx, np.arange(image.shape[2]))]


def main(args=None):
    rclpy.init(args=args)
    node = TransportNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
