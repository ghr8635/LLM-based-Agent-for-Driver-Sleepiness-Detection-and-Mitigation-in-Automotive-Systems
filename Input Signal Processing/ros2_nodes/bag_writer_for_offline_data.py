import rclpy
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import pandas as pd
import os
import cv2
import time

def ros_time_from_unix(t):
    ros_time = Time()
    ros_time.sec = int(t)
    ros_time.nanosec = int((t - int(t)) * 1e9)
    return ros_time

def main():
    # === Setup ===
    image_dir = "ir_images/"
    ir_df = pd.read_csv("ir_timestamps.csv")
    steering_df = pd.read_csv("steering_data.csv")
    lane_df = pd.read_csv("lane_data.csv")

    ir_df['timestamp'] = pd.to_datetime(ir_df['timestamp'])
    steering_df['timestamp'] = pd.to_datetime(steering_df['timestamp'])
    lane_df['timestamp'] = pd.to_datetime(lane_df['timestamp'])

    bridge = CvBridge()

    # === Setup bag writer ===
    writer = SequentialWriter()
    storage_options = StorageOptions(uri='synthetic_bag', storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    writer.open(storage_options, converter_options)

    writer.create_topic({'name': '/ir_camera/image_raw', 'type': 'sensor_msgs/msg/Image', 'serialization_format': 'cdr'})
    writer.create_topic({'name': '/vehicle/steering_angle', 'type': 'std_msgs/msg/Float64', 'serialization_format': 'cdr'})
    writer.create_topic({'name': '/lane/offset', 'type': 'std_msgs/msg/Float64', 'serialization_format': 'cdr'})

    all_msgs = []

    # === Collect all messages ===
    for _, row in ir_df.iterrows():
        ts = row['timestamp'].timestamp()
        filepath = os.path.join(image_dir, row['filename'])
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        msg = bridge.cv2_to_imgmsg(img, encoding='mono8')
        msg.header.stamp = ros_time_from_unix(ts)
        all_msgs.append((ts, '/ir_camera/image_raw', msg))

    for _, row in steering_df.iterrows():
        ts = row['timestamp'].timestamp()
        msg = Float64()
        msg.data = float(row['steering_angle'])
        all_msgs.append((ts, '/vehicle/steering_angle', msg))

    for _, row in lane_df.iterrows():
        ts = row['timestamp'].timestamp()
        msg = Float64()
        msg.data = float(row['lane_offset'])
        all_msgs.append((ts, '/lane/offset', msg))

    # === Sort all messages by time ===
    all_msgs.sort(key=lambda x: x[0])

    # === Write to bag ===
    for ts, topic, msg in all_msgs:
        writer.write(topic, serialize_message(msg), ros_time_from_unix(ts))

    print("[✓] ROS 2 bag written as 'synthetic_bag'")
    
if __name__ == "__main__":
    main()
