#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace ADAM {

void adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M);
}

void adamUpdate(
    torch::Tensor &param,
    torch::Tensor &param_grad,
    torch::Tensor &exp_avg,
    torch::Tensor &exp_avg_sq,
    torch::Tensor &visible,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
);