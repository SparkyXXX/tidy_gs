import os
import collections
import numpy as np
from plyfile import PlyData

SceneInfo = collections.namedtuple("SceneInfo", ["points", "colors", "scene_radius"])

def get_scene_radius(cam_infos):
    cam_positions = []
    for cam in cam_infos:
        W2C = cam.view.w2c_np
        C2W = np.linalg.inv(W2C)
        cam_positions.append(C2W[:3, 3:4])
    cam_positions_stacked = np.hstack(cam_positions)
    cam_center = np.mean(cam_positions_stacked, axis=1, keepdims=True)
    distances = np.linalg.norm(cam_positions_stacked - cam_center, axis=0, keepdims=True)    
    scene_radius = 1.1 * np.max(distances)
    return scene_radius

def assemble_scene_info(path):
    plydata = PlyData.read(os.path.join(path, "sparse/0/points3D.ply"))
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

def packageSceneInfo(path, train_cam_infos):
    scene_radius = get_scene_radius(train_cam_infos)
    points, colors = assemble_scene_info(path)
    scene_info = SceneInfo(points=points, colors=colors, scene_radius=scene_radius)
    return scene_info