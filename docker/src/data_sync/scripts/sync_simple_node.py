import rclpy
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import Image
from data_sync.msg import DrivingInfo, SyncedOutput   #need to create 
from builtin_interfaces.msg import Time
import time

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        # Buffer for driving messages
        self.driving_buffer = deque(maxlen=100)

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.create_subscription(DrivingInfo, '/driving_info', self.driving_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(SyncedOutput, '/synced_output', 10)

        self.get_logger().info("Sync Node initialized and running...")

    def driving_callback(self, msg):
        # Buffer driving messages
        self.driving_buffer.append({
            'timestamp': msg.timestamp,
            'steering_angle': msg.steering_angle,
            'lane_offset': msg.lane_offset
        })

    def camera_callback(self, image_msg):
        # Convert ROS time to float UNIX time
        cam_stamp = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9

        if not self.driving_buffer:
            self.get_logger().warn("No driving data to sync with.")
            return

        # Find closest driving message based on timestamp
        closest = min(self.driving_buffer, key=lambda x: abs(x['timestamp'] - cam_stamp))

        # Build and publish synced message
        out_msg = SyncedOutput()
        out_msg.stamp = image_msg.header.stamp
        out_msg.image = image_msg
        out_msg.steering_angle = closest['steering_angle']
        out_msg.lane_offset = closest['lane_offset']

        self.publisher.publish(out_msg)
        self.get_logger().info(f"Published synced message at camera time {cam_stamp:.3f}")

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