#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from custom_msgs.msg import SyncedData
from collections import deque
from rospy import Time

class SyncNode:
    def __init__(self):
        rospy.init_node('sync_node', anonymous=True)
        self.bridge = CvBridge()

        self.steering_buffer = deque(maxlen=10)
        self.lane_buffer = deque(maxlen=10)
        self.last_steering = None
        self.last_lane = None

        self.sub_camera = rospy.Subscriber('/ir_camera/image_raw', Image, self.ir_callback)
        self.sub_steering = rospy.Subscriber('/vehicle/steering_angle', Float64, self.steering_callback)
        self.sub_lane = rospy.Subscriber('/lane/offset', Float64, self.lane_callback)

        self.pub_synced = rospy.Publisher('/synced_data', SyncedData, queue_size=10)

    def steering_callback(self, msg):
        now = rospy.Time.now()
        self.last_steering = (msg.data, now)
        self.steering_buffer.append((msg.data, now))

    def lane_callback(self, msg):
        now = rospy.Time.now()
        self.last_lane = (msg.data, now)
        self.lane_buffer.append((msg.data, now))

    def ir_callback(self, msg):
        cam_time = rospy.Time.now()

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
        fused_msg.camera_time = cam_time  # rospy.Time is used directly

        self.pub_synced.publish(fused_msg)

def main():
    node = SyncNode()
    rospy.spin()

if __name__ == '__main__':
    main()
