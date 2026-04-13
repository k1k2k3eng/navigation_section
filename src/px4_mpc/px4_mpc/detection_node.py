#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from mpc_msgs.msg import PerceptionEvent, PipelineDetection

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class DetectionNode(Node):

    def __init__(self):
        super().__init__('pipeline_detection_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('detection_topic', '/perception/pipeline_detection')
        self.declare_parameter('perception_event_topic', '/perception/event')
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_class_name', 'crack')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('enable_mock_detection', False)
        self.declare_parameter('mock_confidence', 0.95)

        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.perception_event_topic = self.get_parameter('perception_event_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.target_class_name = self.get_parameter('target_class_name').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.enable_mock_detection = bool(self.get_parameter('enable_mock_detection').value)
        self.mock_confidence = float(self.get_parameter('mock_confidence').value)

        self.bridge = CvBridge() if CvBridge is not None else None
        self.model = None
        self.model_ready = False
        self.warned_missing_runtime = False

        if not self.enable_mock_detection and self.model_path and YOLO is not None:
            self.model = YOLO(self.model_path)
            self.model_ready = True
            self.get_logger().info(f'YOLO model loaded: {self.model_path}')

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(PipelineDetection, self.detection_topic, 10)
        self.perception_event_pub = self.create_publisher(PerceptionEvent, self.perception_event_topic, 10)

    def image_callback(self, msg):
        if self.enable_mock_detection:
            self.publish_mock_detection(msg)
            return

        if not self.model_ready or self.bridge is None:
            if not self.warned_missing_runtime:
                self.get_logger().warn(
                    'Detection runtime not ready. Install cv_bridge and ultralytics, then set model_path, '
                    'or enable_mock_detection for pipeline testing.'
                )
                self.warned_missing_runtime = True
            self.publish_empty_detection(msg)
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model.predict(image, verbose=False, conf=self.conf_threshold)[0]
        detection = self.extract_best_detection(result, msg)
        self.detection_pub.publish(detection)
        self.perception_event_pub.publish(self.perception_event_from_detection(detection))

    def publish_empty_detection(self, msg):
        detection = PipelineDetection()
        detection.stamp = msg.header.stamp
        detection.source_frame_id = msg.header.frame_id
        detection.crack_detected = False
        self.detection_pub.publish(detection)
        self.perception_event_pub.publish(self.perception_event_from_detection(detection))

    def publish_mock_detection(self, msg):
        detection = PipelineDetection()
        detection.stamp = msg.header.stamp
        detection.source_frame_id = msg.header.frame_id
        detection.crack_detected = True
        detection.confidence = self.mock_confidence
        detection.class_name = self.target_class_name
        detection.center_x = 0.5
        detection.center_y = 0.5
        detection.bbox_xmin = 0.35
        detection.bbox_ymin = 0.35
        detection.bbox_xmax = 0.65
        detection.bbox_ymax = 0.65
        self.detection_pub.publish(detection)
        self.perception_event_pub.publish(self.perception_event_from_detection(detection))

    def extract_best_detection(self, result, msg):
        detection = PipelineDetection()
        detection.stamp = msg.header.stamp
        detection.source_frame_id = msg.header.frame_id

        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            detection.crack_detected = False
            return detection

        names = result.names
        best = None
        best_conf = -math.inf
        for box in boxes:
            cls_idx = int(box.cls.item())
            cls_name = self.class_name_from_result(names, cls_idx)
            conf = float(box.conf.item())
            if self.target_class_name and cls_name != self.target_class_name:
                continue
            if conf > best_conf:
                best = box
                best_conf = conf

        if best is None:
            detection.crack_detected = False
            return detection

        xyxy = best.xyxy[0].tolist()
        width = max(float(result.orig_shape[1]), 1.0)
        height = max(float(result.orig_shape[0]), 1.0)
        detection.crack_detected = True
        detection.confidence = float(best.conf.item())
        detection.class_name = self.class_name_from_result(names, int(best.cls.item()))
        detection.bbox_xmin = float(xyxy[0] / width)
        detection.bbox_ymin = float(xyxy[1] / height)
        detection.bbox_xmax = float(xyxy[2] / width)
        detection.bbox_ymax = float(xyxy[3] / height)
        detection.center_x = float((xyxy[0] + xyxy[2]) * 0.5 / width)
        detection.center_y = float((xyxy[1] + xyxy[3]) * 0.5 / height)
        return detection

    def class_name_from_result(self, names, cls_idx):
        if isinstance(names, dict):
            return names.get(cls_idx, str(cls_idx))
        if isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
            return str(names[cls_idx])
        return str(cls_idx)

    def perception_event_from_detection(self, detection):
        event = PerceptionEvent()
        event.stamp = detection.stamp
        event.source_frame_id = detection.source_frame_id
        event.modality = 'vision'
        event.event_type = detection.class_name if detection.class_name else 'crack'
        event.detected = detection.crack_detected
        event.confidence = float(detection.confidence)
        event.scalar_value = float(detection.confidence)
        event.position_valid = False
        event.hold_time_sec = 3.0
        event.recommended_speed = 0.0
        event.center_x = float(detection.center_x)
        event.center_y = float(detection.center_y)
        event.bbox_xmin = float(detection.bbox_xmin)
        event.bbox_ymin = float(detection.bbox_ymin)
        event.bbox_xmax = float(detection.bbox_xmax)
        event.bbox_ymax = float(detection.bbox_ymax)
        return event


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
