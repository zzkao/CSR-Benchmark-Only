#!/bin/bash
# Environment Setup / Requirement / Installation

# Install ROS (assuming Ubuntu - using melodic as default)
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full -y
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install ROS dependencies
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential -y
sudo rosdep init
rosdep update

# Create catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install GTSAM library (Georgia Tech Smoothing and Mapping library, 4.0.0-alpha2)
wget -O ~/Downloads/gtsam.zip https://github.com/borglab/gtsam/archive/4.0.0-alpha2.zip
cd ~/Downloads/ && unzip gtsam.zip -d ~/Downloads/
cd ~/Downloads/gtsam-4.0.0-alpha2/
mkdir build && cd build
cmake ..
sudo make install

# Install additional ROS packages
sudo apt install ros-melodic-pcl-ros ros-melodic-pcl-conversions ros-melodic-cv-bridge ros-melodic-tf ros-melodic-image-transport -y

# Data / Checkpoint / Weight Download (URL)

# Download Jackal dataset (sample bag files)
mkdir -p ~/datasets/jackal_dataset
cd ~/datasets/jackal_dataset
# Note: Manual download required from Google Drive
# https://drive.google.com/drive/folders/1_t5fX5yIqY-y6sAifY8pVWX4O9LCK5R2?usp=sharing

# Download Stevens VLP16 dataset (larger dataset with 20K+ scans)
mkdir -p ~/datasets/stevens_dataset
cd ~/datasets/stevens_dataset
# Note: Manual download required from Google Drive
# https://drive.google.com/drive/folders/16p5UPUCZ1uK0U4XE-hJKjazTsRghEMJa?usp=sharing

# Training
# Note: LeGO-LOAM is not a learning-based method, no training required

# Inference / Demonstration

# Clone and compile LeGO-LOAM
cd ~/catkin_ws/src
git clone https://github.com/RobustFieldAutonomyLab/LeGO-LOAM.git
cd ..
catkin_make -j1

# Source the workspace
source ~/catkin_ws/devel/setup.bash

# Launch LeGO-LOAM system
roslaunch lego_loam run.launch

# In a separate terminal, play bag files with sample data
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# For real-time usage (modify launch file parameter)
# Set /use_sim_time to false in run.launch for real robot usage

# Testing / Evaluation

# Test with sample bag files (assuming downloaded to ~/datasets/)
cd ~/datasets/jackal_dataset
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# Test with Stevens dataset
cd ~/datasets/stevens_dataset
rosbag play *.bag --clock --topic /velodyne_points /imu/data

# Visualize results in RViz (automatically launched with run.launch)
# Results will be displayed in RViz showing:
# - Point cloud segmentation
# - Lidar odometry trajectory
# - Mapping results
# - Ground plane detection

# Performance evaluation (check terminal output for timing information)
# The system outputs real-time pose estimation and mapping results
