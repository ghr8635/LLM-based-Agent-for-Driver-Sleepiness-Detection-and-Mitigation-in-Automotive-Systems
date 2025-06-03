#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import SyncedData, ProcessedData
import cv2
import numpy as np

class PreprocessNode:
    def __init__(self):
        rospy.init_node('preprocess_node', anonymous=True)
        self.bridge = CvBridge()

        self.sub_synced = rospy.Subscriber('/synced_data', SyncedData, self.callback)
        self.pub_processed = rospy.Publisher('/processed_data', ProcessedData, queue_size=10)

    def callback(self, msg):
        # Convert image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)

        # Resize
        resized = cv2.resize(blurred, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize steering/lane assuming known min/max (example values)
        norm_steering = (msg.steering_angle + 540) / 1080
        norm_lane = (msg.lane_offset + 1.5) / 3.0

        # Publish processed message
        out = ProcessedData()
        out.image = self.bridge.cv2_to_imgmsg(resized, encoding='mono8')
        out.steering_angle = norm_steering
        out.lane_offset = norm_lane
        out.camera_time = msg.camera_time
        self.pub_processed.publish(out)

def main():
    node = PreprocessNode()
    rospy.spin()

if __name__ == '__main__':
    main()
