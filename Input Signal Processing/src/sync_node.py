# sync_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from custom_msgs.msg import SyncedData
import time
from collections import deque

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')
        self.bridge = CvBridge()

        self.steering_buffer = deque(maxlen=10)
        self.lane_buffer = deque(maxlen=10)
        self.last_steering = None
        self.last_lane = None

        self.sub_camera = self.create_subscription(Image, '/ir_camera/image_raw', self.ir_callback, 10)
        self.sub_steering = self.create_subscription(Float64, '/vehicle/steering_angle', self.steering_callback, 10)
        self.sub_lane = self.create_subscription(Float64, '/lane/offset', self.lane_callback, 10)

        self.pub_synced = self.create_publisher(SyncedData, '/synced_data', 10)

    def steering_callback(self, msg):
        now = self.get_clock().now().to_msg()
        self.last_steering = (msg.data, now)
        self.steering_buffer.append((msg.data, now))

    def lane_callback(self, msg):
        now = self.get_clock().now().to_msg()
        self.last_lane = (msg.data, now)
        self.lane_buffer.append((msg.data, now))

    def ir_callback(self, msg):
        cam_time = self.get_clock().now()

        # Select steering and lane based on most recent available
        if self.steering_buffer and self.lane_buffer:
            steering = self.steering_buffer[-1][0] if self.last_steering else 0.0
            lane = self.lane_buffer[-1][0] if self.last_lane else 0.0
        else:
            # Fallback to last known
            steering = self.last_steering[0] if self.last_steering else 0.0
            lane = self.last_lane[0] if self.last_lane else 0.0

        # Construct and publish fused message
        fused_msg = SyncedData()
        fused_msg.image = msg
        fused_msg.steering_angle = steering
        fused_msg.lane_offset = lane
        fused_msg.camera_time = cam_time.to_msg()

        self.pub_synced.publish(fused_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
