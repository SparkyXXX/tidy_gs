import os
import struct
import collections
import numpy as np
from PIL import Image
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.graphics_utils import getWorld2View, focal2fov, qvec2rotmat

Intrinsics = collections.namedtuple("Intrinsics", ["fx", "fy", "cx", "cy", "width", "height"])
Extrinsics = collections.namedtuple("Extrinsics", ["qvec_w2c", "tvec_w2c"])
ImageInfos = collections.namedtuple("ImageInfos", ["image_name", "image_object", "resolution"])

class Camera(NamedTuple):
    intr: Intrinsics
    extr: Extrinsics
    image: ImageInfos
    fovx: float
    fovy: float
    R_w2c: np.ndarray
    T_w2c: np.ndarray

# CameraParams = collections.namedtuple("CameraParams", ["intr", "extr", "fovx", "fovy", "R_w2c", "T_w2c"])

class BasicPointCloud(NamedTuple):
    points : np.array
    normals : np.array
    colors : np.array

class SceneInfo(NamedTuple):
    ply_path: str
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            track_length_useless = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems_useless = read_next_bytes(
                fid, num_bytes=8*track_length_useless,
                format_char_sequence="ii"*track_length_useless)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
    return xyzs, rgbs

def read_intrinsics_binary(path):
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

def read_extrinsics_binary(path):
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

def combine_image_infos(image_names_dict, path, model_params):
    image_infos_dict = {}
    for idx in range(len(image_names_dict)):
        image_name = image_names_dict[idx+1]
        image_object = Image.open(os.path.join(path, "images", image_name))
        orig_w, orig_h = image_object.size
        if model_params.resolution_scale == -1:
            resolution_scale = (orig_w / 1600) if (orig_w > 1600) else 1
            resolution = (round(orig_w / resolution_scale), round(orig_h / resolution_scale))
        else:
            resolution = round(orig_w / model_params.resolution_scale), round(orig_h / model_params.resolution_scale)
        image_info = ImageInfos(image_name=image_name, image_object=image_object, resolution=resolution)
        image_infos_dict[idx+1] = image_info
    return image_infos_dict

def packageCamera(path, model_params):
    extrs, image_names = read_extrinsics_binary(path)
    intr = read_intrinsics_binary(path)
    images = combine_image_infos(image_names, path, model_params)
    camera_list = []

    fovx = focal2fov(intr.fx, intr.width)
    fovy = focal2fov(intr.fy, intr.height)
    for key in extrs:
        extr = extrs[key]
        image = images[key]
        R_w2c = qvec2rotmat(extr.qvec_w2c)
        T_w2c = extr.tvec_w2c
        camera = Camera(intr=intr, extr=extr, image=image ,fovx=fovx, fovy=fovy, R_w2c=R_w2c, T_w2c=T_w2c)
        camera_list.append(camera)
    return camera_list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View(cam.R_w2c, cam.T_w2c)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    translate = -center
    radius = diagonal * 1.1
    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, eval, llffhold=8):
    cam_infos = package_camera_params(path)

    if eval:
        cam_names = sorted([image_name for image_name in cam_infos.extr.image_name])
        test_cam_names_set = {name for idx, name in enumerate(cam_names) if idx % llffhold == 0}
    else:
        test_cam_names_set = set()

    train_cam_infos = []
    test_cam_infos = []
    for c in cam_infos:
        if c.extr.image_name in test_cam_names_set:
            test_cam_infos.append(c)
        else:
            train_cam_infos.append(c)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        xyz, rgb = read_points3D_binary(bin_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

class NB(): 
    def __init__(self):
        self.resolution_scale = -1

if __name__ == "__main__":
    MyNB = NB()
    MyCamList = packageCamera("./data/Hub", MyNB)
    print("Done")