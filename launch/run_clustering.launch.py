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
            #arguments = ['-d', os.path.join(bringup, 'resource', 'rviz.rviz')]
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
    ])

