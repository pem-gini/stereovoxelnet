from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import TimerAction

from launch.actions import ExecuteProcess 
from ament_index_python.packages import get_package_share_directory

import os
import yaml

def generate_launch_description():
    bringup = get_package_share_directory('stereovoxelnet')
    simParams = os.path.join(bringup, "params")

    return LaunchDescription([
        #################################################################################
        ### rviz
        #################################################################################
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output="screen",
            arguments = ['-d', os.path.join(bringup, 'resource', 'rviz.rviz')]
        ),
        #################################################################################
        ### StereoVoxelNet Camera Dummy node
        #################################################################################
        Node(
            package='stereovoxelnet',
            executable='cameradummy',
            name='cameradummy',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(simParams,'cameradummy.yml')]
        ),
        #################################################################################
        ### StereoVoxelNet clustering node
        #################################################################################
        Node(
            package='stereovoxelnet',
            executable='stereovoxelnet',
            name='stereovoxelnet',
            output='screen',
            emulate_tty=True,
            parameters=[os.path.join(simParams,'stereovoxel.yml')]
        ),
        #################################################################################
        ### Stereo image processing node
        #################################################################################
        # Node(
        #     package='stereo_image_proc',
        #     executable='disparity_node',
        #     name='disparity_node',
        #     parameters=[
        #         {'approximate_sync': True}
        #     ],
        #     output='screen',
        #     emulate_tty=True,
        # ),
        # ComposableNode(
        #     package='stereo_image_proc',
        #     plugin='stereo_image_proc::DisparityNode',
        #     parameters=[{
        #         'approximate_sync': True,
        #         #'use_system_default_qos': True,
        #         #'disparity_range': 128,
        #         #'texture_threshold': 10,
        #     }],
        #     remappings=[
        #         ('left/image_rect', ['left', '/image_rect']),
        #         ('left/camera_info', ['left', '/camera_info']),
        #         ('right/image_rect', ['right', '/image_rect']),
        #         ('right/camera_info', ['right', '/camera_info']),
        #     ]
        # ),
        Node(
            package='stereovoxelnet',
            executable='world_visualizer',
            name='world_visualizer',
            emulate_tty=True,
            output='screen',
            parameters=[os.path.join(simParams,'world_visualizer.yml')]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_camera',
            arguments = ['--x', '0', '--y', '0', '--z', '2', '--roll', '1.570796', '--pitch', '3.14159265359', '--yaw', '0', '--frame-id', 'base_link', '--child-frame-id', 'camera']
        ),
    ])

