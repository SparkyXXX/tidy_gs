#pragma once

#include <functional>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define CUDA_VERSION 11080
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static std::tuple<int, int> forward(
			std::function<char *(size_t)> geometryBuffer,
			std::function<char *(size_t)> binningBuffer,
			std::function<char *(size_t)> imageBuffer,
			std::function<char *(size_t)> sampleBuffer,
			const int P, int D, int M,
			const float *background,
			const int width, int height,
			const float *means3D,
			const float *dc,
			const float *shs,
			const float *opacities,
			const float *scales,
			const float *rotations,
			const float *viewmatrix,
			const float *projmatrix,
			const float *cam_pos,
			const float tan_fovx, float tan_fovy,
			float *out_color,
			float *depth,
			bool antialiasing,
			int *radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R, int B,
			const float *background,
			const int width, int height,
			const float *means3D,
			const float *dc,
			const float *shs,
			const float *opacities,
			const float *scales,
			const float scale_modifier,
			const float *rotations,
			const float *viewmatrix,
			const float *projmatrix,
			const float *campos,
			const float tan_fovx, float tan_fovy,
			const int *radii,
			char *geom_buffer,
			char *binning_buffer,
			char *image_buffer,
			char *sample_buffer,
			const float *dL_dpix,
			const float *dL_invdepths,
			float *dL_dmean2D,
			float *dL_dconic,
			float *dL_dopacity,
			float *dL_dcolor,
			float *dL_dinvdepth,
			float *dL_dmean3D,
			float *dL_dcov3D,
			float *dL_ddc,
			float *dL_dsh,
			float *dL_dscale,
			float *dL_drot,
			bool antialiasing,
			bool debug);
	};
};


std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	// const torch::Tensor& colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	// const float scale_modifier,
	// const torch::Tensor& cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &dc,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	// const bool prefiltered,
	const bool antialiasing,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	// const torch::Tensor& colors,
	const torch::Tensor &opacities,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	// const torch::Tensor& cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dc,
	const torch::Tensor &sh,
	const torch::Tensor &dL_dout_invdepth,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const int B,
	const torch::Tensor &sampleBuffer,
	const bool antialiasing,
	const bool debug);