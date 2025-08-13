import os
import torch
import torchvision
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from setproctitle import setproctitle
from scene import Scene, GaussianModel
from renderer import render
from utils.general_utils import safe_state
from utils.arguments import ModelParams, PipelineParams, get_combined_args

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.img.origin_img_cuda[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(model_params : ModelParams, pipeline : PipelineParams, iteration : int, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree, optimizer_type="sparse_adam")
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(model_params.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
             render_set(model_params.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    setproctitle("Ruixiang's Work 😆")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = ArgumentParser(description="Testing script parameters")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args = get_combined_args(parser)

    safe_state(args.quiet)
    render_sets(mp.extract(args), pp.extract(args), args.iteration, args.skip_train, args.skip_test)