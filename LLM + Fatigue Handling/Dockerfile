FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# Copy packages from the mono-repo
COPY llm_fatigue_handling ./src/llm_fatigue_handling
COPY requirements.txt .
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN . /opt/ros/humble/setup.sh && colcon build

ENTRYPOINT ["/ros_entrypoint.sh"]

# Optional: use launch file
# CMD ["ros2", "launch", "llm_fatigue_handling", "llm_launch.py"]