# launch/llm_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='llm_fatigue_handling',
            executable='llm_node',
            name='llm_node',
            output='screen',
            parameters=[] 
        ),
    ])