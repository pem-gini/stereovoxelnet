import rclpy
from rclpy.node import Node
import os
import struct
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
point_struct = struct.Struct("<fffBBBB")

class WorldVisualizer(Node):
    def __init__(self):
        super().__init__('world_visualizer')
        self.log("Initialiting Stereovoxel World Visualizer Node...")
        ### ros2 params
        self.declare_parameter("debug", True)
        self.declare_parameter("topic_img_left", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_img_right", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_camera_left", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_camera_right", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_disparity", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_world", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.debug: bool = self.get_parameter("debug").value
        # self.targetFrame: str = self.get_parameter("target_frame").value
        self.topicImgLeft: str = self.get_parameter("topic_img_left").value
        #self.topicImgRight: str = self.get_parameter("topic_img_right").value
        #self.topicCameraLeft: str = self.get_parameter("topic_camera_left").value
        #self.topicCameraRight: str = self.get_parameter("topic_camera_right").value
        self.topicDisparity: str = self.get_parameter("topic_disparity").value
        self.topicWorld: str = self.get_parameter("topic_world").value

        self.bridge = CvBridge()

        self.cameraSub = TimeSynchronizer([
            Subscriber(self, Image, self.topicImgLeft),
            #Subscriber(self, Image, self.topicImgRight),
            Subscriber(self, Image, self.topicDisparity),
        ], 10)

        self.cameraSub.registerCallback(self.imageCallback)
        self.worldPub = self.create_publisher(PointCloud2, self.topicWorld, 1)

    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)
    
    def imageCallback(self, msgImgLeft: Image, msgDisparity: Image):
        self.log('Create point cloud of the world')

        # Parse left image
        frameLeft = self.bridge.imgmsg_to_cv2(msgImgLeft, desired_encoding='bgr8')
        left_img = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2RGB)

        # Parse disparity
        disparity = self.bridge.imgmsg_to_cv2(msgDisparity, desired_encoding='passthrough')
        #disparity = np.array(disparity, dtype=np.float32)[:crop_h,:crop_w] / 256.

        # w, h = left_img.size
        # crop_w, crop_h = 880, 400

        # left_img = left_img[h-crop_h:, w-crop_w:]
        # disparity = disparity[h - crop_h:h, w - crop_w: w]
        depth_rgb = np.asarray(left_img)[:, :, :3]

        # Camera intrinsics and extrinsics
        c_u = 4.556890e+2
        c_v = 1.976634e+2
        f_u = 1.003556e+3
        f_v = 1.003556e+3
        b_x = 0.0
        b_y = 0.0
        baseline = 0.54

        def project_image_to_rect(uv_depth):
            ''' Input: nx3 first two channels are uv, 3rd channel
                    is depth in rect camera coord.
                Output: nx3 points in rect camera coord.
            '''
            n = uv_depth.shape[0]
            x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
            y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
            pts_3d_rect = np.zeros((n, 3))
            pts_3d_rect[:, 0] = x
            pts_3d_rect[:, 1] = y
            pts_3d_rect[:, 2] = uv_depth[:, 2]
            return pts_3d_rect

        def project_image_to_velo(uv_depth):
            pts_3d_rect = project_image_to_rect(uv_depth)
            return pts_3d_rect

        mask = disparity > 0
        depth_gt = f_u * baseline / (disparity + 1. - mask)

        mask = disparity > 0
        rows, cols = depth_gt.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth_gt])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud_gt = project_image_to_velo(points)

        points_rgb = depth_rgb[mask].reshape((-1, 3))
        points_bgr = np.flip(points_rgb.astype(int), axis=1)
        points_alpha = np.full(points_bgr.shape[0], 255, dtype=int)
        
        pc_points = cloud_gt.tolist()
        pc_points_bgra = np.column_stack([points_bgr, points_alpha]).tolist()

        header = Header()
        # header.stamp = init_time
        header.stamp = msgDisparity.header.stamp
        header.frame_id = '/camera'

        # Create a PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
        ]

        buffer = bytearray(point_struct.size * len(pc_points))
        for i, point in enumerate(pc_points):
            point_struct.pack_into(buffer, i * point_struct.size, *point, *(pc_points_bgra[i]))
        
        point_cloud = PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=point_struct.size,
                       row_step=len(buffer),
                       data=buffer)
        self.worldPub.publish(point_cloud)

def main(args=None):
    rclpy.init(args=args)

    node = WorldVisualizer()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
