import rclpy
from rclpy.node import Node
import os
import torch
import torch.nn as nn
import numpy as np
from cv_bridge import CvBridge
import cv2
from .net.datasets.data_io import get_transform
from .net.models.Voxel2D_hie import Voxel2D
from .net.utils.ros_utils import create_cloud
from rclpy.node import Node
from rclpy.publisher import Publisher
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import logging
import coloredlogs
import time
from std_msgs.msg import Header
from message_filters import TimeSynchronizer, Subscriber
from stereo_msgs.msg import DisparityImage

crop_w, crop_h = 880, 400

class DisparityExtractor(Node):
    def __init__(self):
        super().__init__('world_visualizer')
        self.log("Initialiting Disparity Extractor Node...")

        self.create_subscription(DisparityImage, '/disparity', self.disparityCallback, 1)
        self.dispImgPub = self.create_publisher(Image, '/disparity_image', 1)

    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)
    
    def disparityCallback(self, disparity: DisparityImage):
        self.log('Publishing new disparity image')
        self.dispImgPub.publish(disparity.image)
        print(disparity.image.width)
        print(disparity.image.data)
        

def main(args=None):
    rclpy.init(args=args)

    node = DisparityExtractor()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
