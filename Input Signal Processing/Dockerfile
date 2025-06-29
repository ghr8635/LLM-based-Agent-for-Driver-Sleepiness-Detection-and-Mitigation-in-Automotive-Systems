FROM ros:humble-ros-base

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install Python3 tools and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python numpy

# Source ROS setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create workspace and copy your packages
WORKDIR /ros2_ws/src
COPY ./src .

# Build your workspace (including custom_msgs)
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.bash && colcon build

# Source ROS 2 + install/setup.bash on container start
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 run your_package your_node"]
