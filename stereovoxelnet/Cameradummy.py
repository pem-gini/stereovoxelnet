import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.time import Time
import os
import yaml
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
from datetime import datetime, timedelta
from std_msgs.msg import Header
from message_filters import TimeSynchronizer, Subscriber
from stereo_msgs.msg import DisparityImage
import rosbag2_py

class CameraDummyNode(Node):
    def __init__(self):
        super().__init__('cameradummy')
        self.log("Initialiting Stereovoxel Camera Dummy Node...")
        ### ros2 params
        self.declare_parameter("debug", True)
        self.declare_parameter("as_rosbag", False)
        self.declare_parameter("topic_img_left", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_img_right", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_camera_left", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_camera_right", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_disparity", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("img_dir", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("frames_per_second", 2.0)
        self.debug: bool = self.get_parameter("debug").value
        # self.targetFrame: str = self.get_parameter("target_frame").value
        self.topicImgLeft: str = self.get_parameter("topic_img_left").value
        self.topicImgRight: str = self.get_parameter("topic_img_right").value
        self.topicCameraLeft: str = self.get_parameter("topic_camera_left").value
        self.topicCameraRight: str = self.get_parameter("topic_camera_right").value
        self.topicDisparity: str = self.get_parameter("topic_disparity").value
        self.imgDir: str = self.get_parameter("img_dir").value
        self.framesPerSecond: float = self.get_parameter("frames_per_second").value
        self.asRosbag: bool = self.get_parameter("as_rosbag").value

        self.bridge = CvBridge()

        self.pubImgLeft = self.create_publisher(Image, self.topicImgLeft, 1)
        self.pubImgRight = self.create_publisher(Image, self.topicImgRight, 1)
        self.pubDisparity = self.create_publisher(Image, self.topicDisparity, 1)
        self.pubDepthImage = self.create_publisher(Image, 'depth_registered/image_rect', 1)
        self.pubCameraLeft = self.create_publisher(CameraInfo, self.topicCameraLeft, 1)
        self.pubCameraRight = self.create_publisher(CameraInfo, self.topicCameraRight, 1)

        self.imgDirLeft = os.path.join(self.imgDir, 'left')
        self.imgDirRight = os.path.join(self.imgDir, 'right')
        self.imgDirDisparity = os.path.join(self.imgDir, 'disparity')
        self.frames: list[str] = []
        self.framesCnt = 0
        self.currentFrame = 0
        self.indexDataset()

        self.cameraInfos: list[CameraInfo] = []
        self.readConfig()

        if self.asRosbag:
            self.startTime = datetime.now()
            self.writer = rosbag2_py.SequentialWriter()
            storage_options = rosbag2_py._storage.StorageOptions(
                uri='stero_image_bag',
                storage_id='mcap')
            converter_options = rosbag2_py._storage.ConverterOptions('', '')
            self.writer.open(storage_options, converter_options)

            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.topicImgLeft,
                type='sensor_msgs/msg/Image',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.topicImgRight,
                type='sensor_msgs/msg/Image',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.topicDisparity,
                type='sensor_msgs/msg/Image',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.topicCameraLeft,
                type='sensor_msgs/msg/CameraInfo',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
            topic_info = rosbag2_py._storage.TopicMetadata(
                name=self.topicCameraRight,
                type='sensor_msgs/msg/CameraInfo',
                serialization_format='cdr')
            self.writer.create_topic(topic_info)

            for _ in range(self.framesCnt):
                self.publishFrame()
        
        else:
            self.create_timer(1.0 / self.framesPerSecond, self.publishFrame)

    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)

    def readConfig(self):
        configPath = os.path.join(self.imgDir, 'calib.yml')
        with open(configPath, 'r') as configFile:
            config = yaml.safe_load(configFile)

        for i in [1,3]:
            dimension = np.fromstring(config[f'S_rect_10{i}'], count=2, sep=' ')
            cameraInfo = CameraInfo()
            cameraInfo.width = int(dimension[0])
            cameraInfo.height = int(dimension[1])
            cameraInfo.distortion_model = 'plumb_bob'
            cameraInfo.d = np.fromstring(config[f'D_10{i}'], count=5, sep=' ').tolist()
            cameraInfo.k = np.fromstring(config[f'K_10{i}'], count=9, sep=' ').tolist()
            cameraInfo.r = np.fromstring(config[f'R_10{i}'], count=9, sep=' ').tolist()
            cameraInfo.p = np.fromstring(config[f'P_rect_10{i}'], count=12, sep=' ').tolist()

            self.cameraInfos.append(cameraInfo)


    def indexDataset(self):
        imagesLeft = sorted(os.listdir(self.imgDirLeft))
        imagesRight = set(os.listdir(self.imgDirRight))

        for frameLeft in imagesLeft:
            if frameLeft in imagesRight:
                self.frames.append(frameLeft)
        self.framesCnt = len(self.frames)
    
    def publishFrame(self):
        currentFrame = self.currentFrame
        if self.currentFrame < self.framesCnt-1:
            self.currentFrame += 1
        else:
            self.currentFrame = 0

        timestamp = self.get_clock().now().to_msg()
        if self.asRosbag:
            timestamp = Time(seconds=(self.startTime + timedelta(seconds=self.currentFrame)).timestamp()).to_msg()

        frame = self.frames[currentFrame]
        self.log(f'Load frame: {frame} ({currentFrame}/{self.framesCnt}) ({timestamp})')

        # Load .png files as tensors
        imgLeft = cv2.imread(os.path.join(self.imgDirLeft, frame))
        imgRight = cv2.imread(os.path.join(self.imgDirRight, frame))
        imgDisparity = cv2.imread(os.path.join(self.imgDirDisparity, frame.replace('.jpg', '.png')), cv2.IMREAD_UNCHANGED)
        imgDisparity = imgDisparity.astype(float) / 256.

        # To grayscale
        imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

        # Fake the BGR values, because the model is expecting a colored image
        imgLeft = np.repeat(imgLeft, 3).reshape(imgLeft.shape+(3,))
        imgRight = np.repeat(imgRight, 3).reshape(imgRight.shape+(3,))

        # Resize while keeping original ratio and then crop to even pixel
        imgLeft = imgLeft[:400,:880]
        imgRight = imgRight[:400,:880]
        imgDisparity = imgDisparity[:400,:880]

        header = Header()
        header.stamp = timestamp
        header.frame_id = '/camera'
        
        for camera in self.cameraInfos: 
            camera.header = header

        # disparity = DisparityImage()
        # disparity.header = header
        # disparity.f = 365.68
        # disparity.t = 0.12

        #print(disparityData)
        #depthImg = np.round((disparity.f * disparity.t / disparityData))

        imgMsgLeft = self.bridge.cv2_to_imgmsg(imgLeft, encoding='bgr8', header=header)
        imgMsgRight = self.bridge.cv2_to_imgmsg(imgRight, encoding='bgr8', header=header)
        imgMsgDisparity = self.bridge.cv2_to_imgmsg(imgDisparity, encoding='passthrough', header=header)

        #disparity.image = imgMsgDisparity

        # left_cam_info = CameraInfo()
        # left_cam_info.width = imgLeft.shape[1]
        # left_cam_info.height = imgLeft.shape[0]
        # left_cam_info.d = distCoeffs_left.T.tolist()[0]
        # left_cam_info.k = cameraMatrix_left.reshape(-1,9).tolist()[0]
        # left_cam_info.r = R1.reshape(-1,9).tolist()[0]
        # left_cam_info.p = P1.reshape(-1,12).tolist()[0]
        # left_cam_info.distortion_model = "plumb_bob"
        # left_cam_info.header = Header()
        # left_cam_info.header.stamp = timestamp
        # left_cam_info.header.frame_id = "zed_left"

        if self.asRosbag:
            timestamp_nano = int(timestamp.sec * 1e9 + timestamp.nanosec)
            #self.log(f'{timestamp.sec}s + {timestamp.nanosec} ns: {timestamp_nano}')
            self.writer.write(self.topicCameraLeft, serialize_message(self.cameraInfos[0]), timestamp_nano)
            self.writer.write(self.topicCameraRight, serialize_message(self.cameraInfos[1]), timestamp_nano)
            self.writer.write(self.topicImgLeft, serialize_message(imgMsgLeft), timestamp_nano)
            self.writer.write(self.topicImgRight, serialize_message(imgMsgRight), timestamp_nano)
            self.writer.write(self.topicDisparity, serialize_message(imgMsgDisparity), timestamp_nano)
        else:
            self.pubCameraLeft.publish(self.cameraInfos[0])
            self.pubCameraRight.publish(self.cameraInfos[1])

            self.pubImgLeft.publish(imgMsgLeft)
            self.pubImgRight.publish(imgMsgRight)
            self.pubDisparity.publish(imgMsgDisparity)
        # self.pubDepthImage.publish(imgMsgDepth)

        # Compute depthmap and publish point cloud
        

def main(args=None):
    rclpy.init(args=args)

    node = CameraDummyNode()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
