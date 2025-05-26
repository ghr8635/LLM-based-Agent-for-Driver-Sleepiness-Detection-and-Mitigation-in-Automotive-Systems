# preprocess_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import SyncedData, ProcessedData
import cv2
import numpy as np

class PreprocessNode(Node):
    def __init__(self):
        super().__init__('preprocess_node')
        self.bridge = CvBridge()

        self.sub_synced = self.create_subscription(SyncedData, '/synced_data', self.callback, 10)
        self.pub_processed = self.create_publisher(ProcessedData, '/processed_data', 10)

    def callback(self, msg):
        # Convert image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

        # Resize
        resized = cv2.resize(blurred, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize image to [0, 1]
        #norm_image = resized.astype(np.float32) / 255.0

        # Normalize steering/lane assuming known min/max (example values)
        norm_steering = (msg.steering_angle + 540) / 1080
        norm_lane = (msg.lane_offset + 1.5) / 3.0

        # Re-encode processed image
        #norm_image_ros = self.bridge.cv2_to_imgmsg((norm_image * 255).astype(np.uint8), encoding='mono8')

        # Publish processed message
        out = ProcessedData()
        out.image = resized
        out.steering_angle = norm_steering
        out.lane_offset = norm_lane
        out.camera_time = msg.camera_time
        self.pub_processed.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
