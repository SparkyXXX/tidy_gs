import os
import sys
from typing import NamedTuple
import torch.nn as nn
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import RUIXIANG_S_NB_TOOL # type: ignore

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            RUIXIANG_S_NB_TOOL.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, dc, shs, scales, rotations):
        raster_settings = self.raster_settings
        return _RasterizeGaussians.apply(means3D, means2D, dc, shs, opacities, scales, rotations, raster_settings)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward( ctx, means3D, means2D, dc, sh, opacities, scales, rotations, raster_settings):
        args = (
            raster_settings.bg,
            means3D,
            opacities,
            scales,
            rotations,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = RUIXIANG_S_NB_TOOL.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, invdepths, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = RUIXIANG_S_NB_TOOL.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(means3D, scales, rotations, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        means3D, scales, rotations, radii, dc, sh, opacities, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                # colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                # cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                dc,
                sh,
                grad_out_depth,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_opacities, grad_means3D, grad_dc, grad_sh, grad_scales, grad_rotations = RUIXIANG_S_NB_TOOL.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_opacities, grad_means3D, grad_dc, grad_sh, grad_scales, grad_rotations = RUIXIANG_S_NB_TOOL.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_dc,
            grad_sh,
            # grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            # grad_cov3Ds_precomp,
            None,
        )

        return grads