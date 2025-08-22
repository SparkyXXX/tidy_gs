#pragma once

#include <torch/extension.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__global__ void preprocessCUDA(int P, int D, int M,
							   const float *orig_points,
							   const glm::vec3 *scales,
							   const glm::vec4 *rotations,
							   const float *opacities,
							   const float *dc,
							   const float *shs,
							   bool *clamped,
							   const float *viewmatrix,
							   const float *projmatrix,
							   const glm::vec3 *cam_pos,
							   const int W, int H,
							   const float tan_fovx, float tan_fovy,
							   const float focal_x, float focal_y,
							   int *radii,
							   glm::vec2 *points_xy_image,
							   float *depths,
							   float *cov3Ds,
							   float *rgb,
							   glm::vec4 *conic_opacity,
							   const dim3 grid,
							   uint32_t *tiles_touched,
							   bool antialiasing);

__global__ void __launch_bounds__(256) renderCUDA(
	const uint2 *__restrict__ ranges,
	const uint32_t *__restrict__ point_list,
	const uint32_t *__restrict__ per_tile_bucket_offset, uint32_t *__restrict__ bucket_to_tile,
	float *__restrict__ sampled_T, float *__restrict__ sampled_ar, float *__restrict__ sampled_ard,
	int W, int H,
	const glm::vec2 *__restrict__ points_xy_image,
	const float *__restrict__ features,
	const glm::vec4 *__restrict__ conic_opacity,
	float *__restrict__ final_T,
	uint32_t *__restrict__ n_contrib,
	uint32_t *__restrict__ max_contrib,
	const float *__restrict__ bg_color,
	float *__restrict__ out_color,
	const float *__restrict__ depths,
	float *__restrict__ invdepth);