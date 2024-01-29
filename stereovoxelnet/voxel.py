from __future__ import print_function, division
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
import struct
from std_msgs.msg import Header
from message_filters import TimeSynchronizer, Subscriber

coloredlogs.install(level="DEBUG")
torch.backends.cudnn.benchmark = True
CURR_DIR = '/home/finn/pem/duckbrain_umbrella/ros2_ws/src/stereovoxelnet/stereovoxelnet'
point_struct = struct.Struct("<fffBBBB")

class StereoVoxelnetNode(Node):
    def __init__(self):
        super().__init__('stereovoxelnet')
        self.log("Initializing Stereovoxel Clustering Node...")
        ### ros2 params
        self.declare_parameter("debug", True)
        self.declare_parameter("target_frame", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_img_left", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_img_right", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_output", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("f_u", 365.68)
        self.declare_parameter("baseline", 0.12)
        self.declare_parameter("max_disparity", 192)
        self.debug: bool = self.get_parameter("debug").value
        self.targetFrame: str = self.get_parameter("target_frame").value
        self.topicImgLeft: str = self.get_parameter("topic_img_left").value
        self.topicImgRight: str = self.get_parameter("topic_img_right").value
        self.topicOut: str = self.get_parameter("topic_output").value
        self.f_u: float = self.get_parameter("f_u").value
        self.baseline: float = self.get_parameter("baseline").value
        self.maxDisparity: int = self.get_parameter("max_disparity").value
        # ###

        self.bridge = CvBridge()
        self.level_i = 3

        # Sync both images
        self.cloudSub = TimeSynchronizer([
            Subscriber(self, Image, self.topicImgLeft),
            Subscriber(self, Image, self.topicImgRight),
            #self.create_subscription(PointCloud2, self.topicImgLeft, self.singleImageCallback, 1),
            #self.create_subscription(PointCloud2, self.topicImgRight, self.singleImageCallback, 1),
        ], 10)
        self.cloudSub.registerCallback(self.imageCallback)
        self.cloudPub: 'list[Publisher]' = []
        for i in range(4):
            self.cloudPub.append(self.create_publisher(PointCloud2, f'{self.topicOut}_{i}', 1))

        self.loadModel()

        self.vox_cost_vol_disps = []
        self.calcDisp()

    def loadModel(self):
        self.log("start loading model")
        voxel_model = Voxel2D(self.maxDisparity, "voxel")
        voxel_model = nn.DataParallel(voxel_model)
        if torch.cuda.is_available():
            self.log('Cuda enabled for stereo voxel predictions!')
            voxel_model.cuda()
        ckpt_path = os.path.join(CURR_DIR, "net/voxel.ckpt")
        self.log("model {} loaded".format(ckpt_path))
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location="cuda")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        voxel_model.load_state_dict(state_dict['model'])
        self.log("model weight loaded")
        self.model = voxel_model
        self.model.eval()

    def calcDisp(self):
        f_u = 1.003556e+3
        baseline = 0.54

        # calculate voxel cost volume disparity set
        vox_cost_vol_disp_set = set()
        max_disp = 192
        # depth starting from voxel_size since 0 will cause issue
        for z in np.arange(0.5, 32, 2.0):
            # get respective disparity
            d = f_u * baseline / z

            if d > max_disp:
                continue

            # real disparity -> disparity in feature map
            vox_cost_vol_disp_set.add(round(d/4))

        vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        vox_cost_vol_disps = sorted(vox_cost_vol_disps)
        # vox_cost_vol_disps = vox_cost_vol_disps[1:]

        tmp = []
        for i in vox_cost_vol_disps:
            tmp.append(torch.unsqueeze(torch.Tensor([i]), 0))
        self.vox_cost_vol_disps = tmp
        self.log("Disparity level calculated")
    
    def pub_empty_pc(self, level_i):
        header = Header()
        # header.stamp = init_time
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.targetFrame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),                
                ]
        pc = create_cloud(header, fields, [])

        for i in range(4):
            if i != level_i:
                self.cloudPub[i].publish(pc)

    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)

    def singleImageCallback(self, msg):
        pass

    def predictionToPointcloud(self, vox_pred, grid_size):
        vox_pred = vox_pred.detach().cpu().numpy()

        voxel_size = 32 / grid_size
        
        offsets = np.array([int(16/voxel_size), int(31/voxel_size), 0])
        mask = np.where(vox_pred >= 0.5)
        xyz_pred = np.asarray(mask) # get back indexes of populated voxels
        cloud = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])

        self.log(f"Size of point cloud: {len(cloud)}, Level {self.level_i}")


        points = cloud.tolist()
        header = Header()
        # header.stamp = init_time
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.targetFrame

        # Create a PointCloud2 message
        # fields = [
        #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # ]

                # Create a PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
        ]

        buffer = bytearray(point_struct.size * len(points))
        weights = vox_pred[mask]
        for i, point in enumerate(points):
            weight = weights[i]
            color = [0, 0, 0, 0] # bgra
            if weight >= 0.9:
                color = [0,   0, 255, 255] # red
            elif weight >= 0.8:
                color = [0, 150, 255, 255] # orange
            elif weight >= 0.7:
                color = [0, 255, 255, 255] # yellow
            elif weight >= 0.6:
                color = [0, 255, 150, 255] # 
            elif weight >= 0.5:
                color = [0, 255,   0, 255]

            point_struct.pack_into(buffer, i * point_struct.size, *point, *color)
        
        point_cloud = PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=False,
                       is_bigendian=False,
                       fields=fields,
                       point_step=point_struct.size,
                       row_step=len(buffer),
                       data=buffer)
        
        return point_cloud # create_cloud(header, fields, points)


    def imageCallback(self, msgImgLeft: Image, msgImgRight: Image):
        if self.model is None:
            self.warning("Model is not ready")
            return
        # Parse left image
        frameLeft = self.bridge.imgmsg_to_cv2(msgImgLeft, desired_encoding='bgr8')
        frameLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2RGB)

        # Parse right image
        frameRight = self.bridge.imgmsg_to_cv2(msgImgRight, desired_encoding='bgr8')
        frameRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2RGB)

        # w, h = frameLeft.size
        # crop_w, crop_h = 880, 400

        # frameLeft = frameLeft.crop((w - crop_w, h - crop_h, w, h))
        # frameRight = frameRight.crop((w - crop_w, h - crop_h, w, h))

        # Prepare images
        processed = get_transform()
        
        sample_left = processed(frameLeft)
        sample_right = processed(frameRight)

        # sample_left = torch.Tensor(sample_left)
        # sample_right = torch.Tensor(sample_right)

        sample_left = torch.unsqueeze(sample_left, dim=0)
        sample_right = torch.unsqueeze(sample_right, dim=0)

        print(f'Img shape: {sample_left.shape}')

        with torch.no_grad():
            self.log('Run prediction')
            voxel_sizes = [8,16, 32, 64]
            vox_pred = self.model(sample_left.cuda(), sample_right.cuda(), self.vox_cost_vol_disps)[0]
            # size_len = vox_pred[self.level_i][0].shape[0]
            # if torch.all(vox_pred[self.level_i][0][:,:,:int(size_len*0.5)] < 0.5) and self.level_i > 0:
            #     self.level_i -= 1
            # elif self.level_i < 3:
            #     self.level_i += 1
            self.level_i = 3
            assert(self.level_i >= 0 and self.level_i <= 3)
            vox_pred_level = vox_pred[self.level_i][0]
            print(f'Predicted level {self.level_i}')
            pc = self.predictionToPointcloud(vox_pred_level,voxel_sizes[self.level_i])
            
            self.cloudPub[self.level_i].publish(pc)
            self.pub_empty_pc(self.level_i)

        # self.error("%s" % (time.time() - start))
