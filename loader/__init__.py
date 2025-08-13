import os
import random
from loader.load_cameras import packageCameras, separateCamerasToTrainTest
from loader.load_scene import packageSceneInfo, SceneInfo
from loader.gaussian_model import GaussianModel
from utils.general_utils import searchForMaxIteration

class AllData:
    train_cam_list: list
    test_cam_list: list
    scene_info: SceneInfo
    gaussians : GaussianModel
    model_path: str
    def __init__(self, train_cam_list, test_cam_list, scene_info, gaussians, model_path):
        self.train_cam_list = train_cam_list
        self.test_cam_list = test_cam_list
        self.scene_info = scene_info
        self.gaussians = gaussians
        self.model_path = model_path

def packageAllData(model_params, load_iteration=None, shuffle=True):
    cam_list = packageCameras(model_params)
    train_cam_list, test_cam_list = separateCamerasToTrainTest(cam_list, model_params.eval)
    scene_info = packageSceneInfo(model_params.source_path, train_cam_list)
    gaussians = GaussianModel(model_params.sh_degree, optimizer_type="sparse_adam")
    
    if shuffle:
        random.shuffle(train_cam_list)
        random.shuffle(test_cam_list)

    loaded_iter = None
    if load_iteration:
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_params.model_path, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(loaded_iter))
    if loaded_iter:
        gaussians.load_ply(os.path.join(model_params.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene_info, scene_info.scene_radius)
    
    all_data = AllData(train_cam_list=train_cam_list, test_cam_list=test_cam_list, 
                  scene_info=scene_info, gaussians=gaussians, model_path=model_params.source_path)
    return all_data