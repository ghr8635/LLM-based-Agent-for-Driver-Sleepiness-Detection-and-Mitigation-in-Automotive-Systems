import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from collections import deque
from custom_msgs.msg import SyncedOutput  # Assuming this is your custom msg

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')
        self.bridge = CvBridge()

        # Buffers and state
        self.image_buffer = deque(maxlen=10)
        self.latest_driving_data = None

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Vector3, '/driving_info', self.driving_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(SyncedOutput, '/synced_output', 10)

        self.get_logger().info("Sync Node with Vector3 driving info is running.")

    def camera_callback(self, msg):
        now = self.get_clock().now().to_msg()
        self.image_buffer.append((msg, now))

    def driving_callback(self, msg):
        # Store latest values
        self.latest_driving_data = {
            'steering_angle': msg.steering_angle,
            'lane_offset': msg.lane_offset
        }

        # Proceed only if we have at least one image
        if not self.image_buffer:
            self.get_logger().warn("No image available to sync with driving info.")
            return

        # Get the latest image
        image_msg, image_time = self.image_buffer[-1]

        # Build and publish fused message
        fused_msg = SyncedOutput()
        fused_msg.image = image_msg
        fused_msg.steering_angle = self.latest_driving_data['steering_angle']
        fused_msg.lane_offset = self.latest_driving_data['lane_offset']
        fused_msg.camera_time = image_time  # From PC time at image reception

        self.publisher.publish(fused_msg)
        self.get_logger().info("Published synced data.")

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
