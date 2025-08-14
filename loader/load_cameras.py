import os
import torch
import collections
import numpy as np
from torch import nn
from PIL import Image
from utils.general_utils import PILtoTorch, read_next_bytes
from utils.graphics_utils import focal2fov, qvec2rotmat, getWorld2View, getProjectionMatrix

Intrinsics = collections.namedtuple("Intrinsics", ["fx", "fy", "cx", "cy", "width", "height"])
Extrinsics = collections.namedtuple("Extrinsics", ["qvec_w2c", "tvec_w2c"])
Viewpoints = collections.namedtuple("Viewpoints", ["fovx", "fovy", "w2c_np", "w2c_cuda", "proj_cuda", "whole_transform_cuda", "cam_center_cuda"])
ImageInfos = collections.namedtuple("ImageInfos", ["img_name", "img_width", "img_height", "img_object_pil", "origin_img_cuda"])

class Camera(nn.Module):
    intr: Intrinsics
    extr: Extrinsics
    view: Viewpoints
    img: ImageInfos
    def __init__(self, intr, extr, view, img):
        super(Camera, self).__init__()
        self.intr = intr
        self.extr = extr
        self.view = view
        self.img = img

def assemble_intrinsics(path):
    num_params = 4
    with open(os.path.join(path, "sparse/0/cameras.bin"), "rb") as fid:
        num_cameras_useless = read_next_bytes(fid, 8, "Q")[0]
        camera_properties_useless = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
        camera_id_useless = camera_properties_useless[0]
        model_id_useless = camera_properties_useless[1]
        width = camera_properties_useless[2]
        height = camera_properties_useless[3]
        intrinsics = np.array(read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params))
        intrinsics_tuple = Intrinsics(fx=intrinsics[0], fy=intrinsics[1], cx=intrinsics[2], cy=intrinsics[3], width=width, height=height);
    return intrinsics_tuple

def assemble_extrinsics(path):
    extrinsics_dict = {}
    image_names_dict = {}
    with open(os.path.join(path, "sparse/0/images.bin"), "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_idx = binary_image_properties[0]
            qvec_w2c = np.array(binary_image_properties[1:5])
            tvec_w2c = np.array(binary_image_properties[5:8])
            camera_id_useless = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D_useless = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s_useless = read_next_bytes(fid, num_bytes=24*num_points2D_useless, format_char_sequence="ddq"*num_points2D_useless)
            xys_useless = np.column_stack([tuple(map(float, x_y_id_s_useless[0::3])), tuple(map(float, x_y_id_s_useless[1::3]))])
            point3D_ids_useless = np.array(tuple(map(int, x_y_id_s_useless[2::3])))
            extrinsics_dict[image_idx] = Extrinsics(qvec_w2c=qvec_w2c, tvec_w2c=tvec_w2c)
            image_names_dict[image_idx] = image_name
    return extrinsics_dict, image_names_dict

def assemble_viewpoints(intr, extrs):
    viewpoints_dict = {}
    
    zfar = 100.0
    znear = 0.01
    fovx = focal2fov(intr.fx, intr.width)
    fovy = focal2fov(intr.fy, intr.height)
    proj_cuda = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()

    for key in extrs:
        extr = extrs[key]
        R_w2c = qvec2rotmat(extr.qvec_w2c)
        T_w2c = extr.tvec_w2c
        w2c_np = getWorld2View(R_w2c, T_w2c)
        w2c_cuda = torch.tensor(w2c_np).transpose(0, 1).cuda()
        cam_center_cuda = w2c_cuda.inverse()[3, :3]
        whole_transform_cuda = (w2c_cuda.unsqueeze(0).bmm(proj_cuda.unsqueeze(0))).squeeze(0)

        viewpoint = Viewpoints(fovx=fovx, fovy=fovy, w2c_np=w2c_np, w2c_cuda=w2c_cuda, proj_cuda=proj_cuda, 
                              whole_transform_cuda=whole_transform_cuda, cam_center_cuda=cam_center_cuda)
        viewpoints_dict[key] = viewpoint
    return viewpoints_dict

def assemble_image_infos(image_names_dict, model_params):
    image_infos_dict = {}
    for idx in range(len(image_names_dict)):
        img_name = image_names_dict[idx+1]
        img_object_pil = Image.open(os.path.join(model_params.source_path, "images", img_name))
        orig_w, orig_h = img_object_pil.size
        if model_params.resolution_scale == -1:
            resolution_scale = (orig_w / 1600) if (orig_w > 1600) else 1
            resolution = (round(orig_w / resolution_scale), round(orig_h / resolution_scale))
        else:
            resolution = round(orig_w / model_params.resolution_scale), round(orig_h / model_params.resolution_scale)

        resized_img_object_torch = PILtoTorch(img_object_pil, resolution)
        gt_img_torch = resized_img_object_torch[:3, ...]
        origin_img_cuda = gt_img_torch.clamp(0.0, 1.0).cuda()
        img_width = origin_img_cuda.shape[2]
        img_height = origin_img_cuda.shape[1]

        image_infos = ImageInfos(img_name=img_name, img_width=img_width, img_height=img_height,
                                 img_object_pil=img_object_pil, origin_img_cuda=origin_img_cuda)
        image_infos_dict[idx+1] = image_infos
    return image_infos_dict

def packageCameras(model_params):
    intr  = assemble_intrinsics(model_params.source_path)
    extrs, img_names = assemble_extrinsics(model_params.source_path)
    views = assemble_viewpoints(intr, extrs)
    imgs = assemble_image_infos(img_names, model_params)

    cam_list = []
    for key in extrs:
        extr = extrs[key]
        view = views[key]
        img = imgs[key]
        cam = Camera(intr=intr, extr=extr, view=view, img=img)
        cam_list.append(cam)
    return cam_list

# hrx: set llffhold to 100 temporarily!!!
def separateCamerasToTrainTest(cam_list, eval, llffhold=100):
    if eval:
        img_names = sorted([cam.img.img_name for cam in cam_list])
        test_img_names = {name for idx, name in enumerate(img_names) if idx % llffhold == 0}
    else:
        test_img_names = set()

    train_cam_list = []
    test_cam_list = []
    for cam in cam_list:
        if cam.img.img_name in test_img_names:
            test_cam_list.append(cam)
        else:
            train_cam_list.append(cam)
    return train_cam_list, test_cam_list