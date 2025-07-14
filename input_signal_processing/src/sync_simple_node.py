import rclpy
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from data_sync.msg import SyncedOutput  # You still need this for combined output

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        self.latest_driving_data = None

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(Vector3, '/driving_info', self.driving_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(SyncedOutput, '/synced_output', 10)

        self.get_logger().info("Sync Node with Vector3 driving info is running.")

    def driving_callback(self, msg):
        # Store latest values
        self.latest_driving_data = {
            'steering_angle': msg.x,
            'lane_offset': msg.y
        }

    def camera_callback(self, image_msg):
        if self.latest_driving_data is None:
            self.get_logger().warn("No driving data received yet.")
            return

        # Build and publish output
        out_msg = SyncedOutput()
        out_msg.stamp = image_msg.header.stamp
        out_msg.image = image_msg
        out_msg.steering_angle = self.latest_driving_data['steering_angle']
        out_msg.lane_offset = self.latest_driving_data['lane_offset']

        self.publisher.publish(out_msg)
        self.get_logger().info("Published synced message with camera + driving data.")

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
