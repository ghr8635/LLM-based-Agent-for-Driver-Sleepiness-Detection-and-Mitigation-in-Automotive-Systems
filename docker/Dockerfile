FROM ros:humble-ros-base

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Python3 tools and ROS packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-builtin-interfaces \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python numpy

# Source ROS setup in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create workspace and copy your package into container
WORKDIR /ros2_ws
COPY ./src ./src

# Build your workspace (after sourcing ROS environment)
RUN source /opt/ros/humble/setup.bash && colcon build

# Set entrypoint: source ROS and workspace, then run your node
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 run data_sync sync_node.py"]
