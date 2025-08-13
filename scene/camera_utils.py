import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View, getProjectionMatrix, fov2focal

class Camera(nn.Module):
    def __init__(self, resolution, R, T, FoVx, FoVy, image, image_name,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(Camera, self).__init__()
        # self.uid = uid
        # self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].cuda()
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].cuda())

        self.original_image = gt_image.clamp(0.0, 1.0).cuda()
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        # TODO: 这里的转置，是不是因为CUDA中是列主序？
        self.world_view_transform = torch.tensor(getWorld2View(R, T, trans, scale)).transpose(0, 1).cuda()
        temp_projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.projection_matrix = (self.world_view_transform.unsqueeze(0).bmm(temp_projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def cameraList_from_camInfos(cam_infos, model_params):
    camera_list = []
    for cam_info in cam_infos:
        camera = Camera(resolution, R=cam_info.R_w2c, T=cam_info.T_w2c, FoVx=cam_info.fovx, FoVy=cam_info.fovy,
                  image=image, image_name=cam_info.extr.image_name)
        camera_list.append(camera)
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry