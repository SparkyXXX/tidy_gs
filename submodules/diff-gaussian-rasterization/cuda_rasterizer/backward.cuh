#pragma once

#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#define CUDA_VERSION 11080
#include <glm/glm.hpp>

__global__ void computeCov2DCUDA(int P,
                                 const glm::vec3 *means,
                                 const int *radii,
                                 const float *cov3Ds,
                                 const float h_x, float h_y,
                                 const float tan_fovx, float tan_fovy,
                                 const float *view_matrix,
                                 const float *opacities,
                                 const float *dL_dconics,
                                 float *dL_dopacity,
                                 const float *dL_dinvdepth,
                                 glm::vec3 *dL_dmeans,
                                 float *dL_dcov,
                                 bool antialiasing);

__global__ void preprocessCUDABackward(
    int P, int D, int M,
    const glm::vec3 *means,
    const int *radii,
    const float *dc,
    const float *shs,
    const bool *clamped,
    const glm::vec3 *scales,
    const glm::vec4 *rotations,
    const float scale_modifier,
    const float *proj,
    const glm::vec3 *campos,
    const glm::vec3 *dL_dmean2D,
    glm::vec3 *dL_dmeans,
    float *dL_dcolor,
    float *dL_dcov3D,
    float *dL_ddc,
    float *dL_dsh,
    glm::vec3 *dL_dscale,
    glm::vec4 *dL_drot,
    float *dL_dopacity);

__global__ void
PerGaussianRenderCUDA(
    const uint2 *__restrict__ ranges,
    const uint32_t *__restrict__ point_list,
    int W, int H, int B,
    const uint32_t *__restrict__ per_tile_bucket_offset,
    const uint32_t *__restrict__ bucket_to_tile,
    const float *__restrict__ sampled_T, const float *__restrict__ sampled_ar, const float *__restrict__ sampled_ard,
    const float *__restrict__ bg_color,
    const glm::vec2 *__restrict__ points_xy_image,
    const glm::vec4 *__restrict__ conic_opacity,
    const float *__restrict__ colors,
    const float *__restrict__ depths,
    const float *__restrict__ final_Ts,
    const uint32_t *__restrict__ n_contrib,
    const uint32_t *__restrict__ max_contrib,
    const float *__restrict__ pixel_colors,
    const float *__restrict__ pixel_invDepths,
    const float *__restrict__ dL_dpixels,
    const float *__restrict__ dL_invdepths,
    glm::vec3 *__restrict__ dL_dmean2D,
    glm::vec4 *__restrict__ dL_dconic2D,
    float *__restrict__ dL_dopacity,
    float *__restrict__ dL_dcolors,
    float *__restrict__ dL_dinvdepths);