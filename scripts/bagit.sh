#!/usr/bin/env sh

#start recording a new bag
rosbag record /thin_visensor_node/camera_left/camera_info /thin_visensor_node/camera_left/image_raw /thin_visensor_node/camera_left/pose_to_imu_adis16448 /thin_visensor_node/camera_right/camera_info /thin_visensor_node/camera_right/image_raw /thin_visensor_node/camera_right/pose_to_imu_adis16448 /thin_visensor_node/imu_adis16448 /thin_visensor_node/magnetic_field /thin_visensor_node/pressure -o "${HOME}/vibag"

