#pragma once

#include <torch/extension.h>

std::tuple<int, int> rasterizeForward(
	std::function<char *(size_t)> geometryBuffer,
	std::function<char *(size_t)> binningBuffer,
	std::function<char *(size_t)> imageBuffer,
	std::function<char *(size_t)> sampleBuffer,
	const int num_points,
	const float *positions3d,
	const float *scales,
	const float *rotations,
	int degree,
	int max_coef,
	const float *dcs,
	const float *shs,
	const float *background,
	const float *opacities,

	const float *view_matrix,
	const float *projection_matrix,
	const float *camera_poses,
	const int width,
	const int height,
	const float tan_fovx,
	const float tan_fovy,
	float *out_color,
	bool antialiasing,
	int *radii);