#!/bin/bash
set -e

source "/opt/ros/humble/setup.bash"

if [ -f "/ros2_ws/install/setup.bash" ]; then
  source "/ros2_ws/install/setup.bash"
fi

ros2 launch llm_fatigue_handling llm_launch.py 