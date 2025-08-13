import os
import random
from scene.camera_loader import readColmapSceneInfo, separateCameraToTrainTest, packageCameras
from scene.gaussian_model import GaussianModel
from utils.arguments import ModelParams
from utils.general_utils import searchForMaxIteration

class Scene:
    gaussians : GaussianModel
    def __init__(self, model_params : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True):
        self.model_path = model_params.model_path
        self.loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        cam_list = packageCameras(model_params.source_path, model_params)
        self.train_cameras, self.test_cameras = separateCameraToTrainTest(cam_list, model_params.eval)
        scene_info = readColmapSceneInfo(model_params.source_path, self.train_cameras)
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        if shuffle:
            random.shuffle(self.train_cameras)
            random.shuffle(self.test_cameras)
        
        self.gaussians = gaussians
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
