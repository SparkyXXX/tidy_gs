import os
import random
from scene.load_cameras import packageCameras, separateCamerasToTrainTest
from scene.load_scene import packageSceneInfo, SceneInfo
from scene.gaussian_model import GaussianModel
from utils.arguments import ModelParams
from utils.general_utils import searchForMaxIteration

class Whole:
    train_cam_list: list
    test_cam_list: list
    scene_info: SceneInfo
    gaussians : GaussianModel
    model_path: str
    loaded_iter: int
    def __init__(self, train_cam_list, test_cam_list, scene_info, gaussians, model_params, load_iteration=None):
        self.train_cam_list = train_cam_list
        self.test_cam_list = test_cam_list
        self.scene_info = scene_info
        self.gaussians = gaussians
        self.model_path = model_params.model_path
        random.shuffle(train_cam_list)
        random.shuffle(test_cam_list)

        self.loaded_iter = None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info, self.scene_info.scene_radius)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def packageWhole(model_params: ModelParams):
    cam_list = packageCameras(model_params)
    train_cam_list, test_cam_list = separateCamerasToTrainTest(cam_list, model_params.eval)
    scene_info = packageSceneInfo(model_params.source_path, train_cam_list)
    gaussians = GaussianModel(model_params.sh_degree, optimizer_type="sparse_adam")
    whole = Whole(train_cam_list=train_cam_list, test_cam_list=test_cam_list, 
                  scene_info=scene_info, gaussians=gaussians, model_params=model_params)
    return whole