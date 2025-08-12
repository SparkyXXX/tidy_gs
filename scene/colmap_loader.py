import os
import struct
import collections
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from utils.graphics_utils import getWorld2View, focal2fov

class BasicPointCloud(NamedTuple):
    points : np.array
    normals : np.array
    colors : np.array

Intrinsics = collections.namedtuple("Intrinsics", ["fx", "fy", "cx", "cy", "width", "height"])
Extrinsics = collections.namedtuple("Extrinsics", ["qvec", "tvec", "image_name"])

class CameraInfo(NamedTuple):
    intr: Intrinsics
    extr: Extrinsics
    fovx: float
    fovy: float
    R: np.array
    T: np.array
    is_test: bool

# class ImageInfo(NamedTuple):
#     camera_params: CameraInfo


class SceneInfo(NamedTuple):
    ply_path: str
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

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

def read_intrinsics_binary(path_to_model_file):
    num_params = 4
    with open(path_to_model_file, "rb") as fid:
        num_cameras_useless = read_next_bytes(fid, 8, "Q")[0]
        camera_properties_useless = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
        camera_id_useless = camera_properties_useless[0]
        model_id_useless = camera_properties_useless[1]
        width = camera_properties_useless[2]
        height = camera_properties_useless[3]
        params = np.array(read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params))
    return Intrinsics(fx=params[0], fy=params[1], cx=params[2], cy=params[3], width=width, height=height)

def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_idx = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
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
            images[image_idx] = Extrinsics(qvec=qvec, tvec=tvec, image_name=image_name)
    return images

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
        W2C = getWorld2View(cam.R, cam.T)
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

def packageCameraInfos(path):    
    extrs = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
    intr = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    fovx = focal2fov(intr.fx, intr.width)
    fovy = focal2fov(intr.fy, intr.height)
    cam_infos = []
    for _, key in enumerate(extrs):
        extr = extrs[key]
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        cam_info = CameraInfo(intr=intr, extr=extr, fovx=fovx, fovy=fovy, R=R, T=T, is_test=False)        # cam_info = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX,
        cam_infos.append(cam_info)
    return cam_infos

def readColmapSceneInfo(path, eval, train_test_exp, llffhold=8):
    cam_infos = packageCameraInfos(path)

    if eval:
        cam_names = [cam_infos[cam_id].extr.image_name for cam_id in cam_infos.extr]
        cam_names = sorted(cam_names)
        test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
    else:
        test_cam_names_list = []

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
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

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
}