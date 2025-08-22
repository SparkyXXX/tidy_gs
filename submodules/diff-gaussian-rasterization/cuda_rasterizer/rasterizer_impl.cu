#include "auxiliary.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include "rasterizer_impl.cuh"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

std::tuple<int, int> CudaRasterizer::Rasterizer::forward(
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
	float *invdepth,
	bool antialiasing,
	int *radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char *chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	preprocessCUDA<<<(P + 255) / 256, 256>>>(
		P, D, M,
		means3D,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		opacities,
		dc,
		shs,
		geomState.clamped,
		viewmatrix,
		projmatrix,
		(glm::vec3 *)cam_pos,
		width, height,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		antialiasing);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char *binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicate_with_keys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.point_offsets,
		geomState.depths,
		radii,
		tile_grid,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted)
		CHECK_CUDA(, debug)

	int bit = get_higher_msb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				   binningState.list_sorting_space,
				   binningState.sorting_size,
				   binningState.point_list_keys_unsorted, binningState.point_list_keys,
				   binningState.point_list_unsorted, binningState.point_list,
				   num_rendered, 0, 32 + bit),
			   debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// bucket count
	int num_tiles = tile_grid.x * tile_grid.y;
	per_tile_bucket_count<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)
	unsigned int bucket_sum;
	CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);
	// create a state to store. size is number is the total number of buckets * block_size
	size_t sample_chunk_size = required<SampleState>(bucket_sum);
	char *sample_chunkptr = sampleBuffer(sample_chunk_size);
	SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);

	// Let each tile blend its range of Gaussians independently in parallel
	const float *feature_ptr = geomState.rgb;
	renderCUDA<<<tile_grid, block>>>(
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets, sampleState.bucket_to_tile,
		sampleState.T, sampleState.ar, sampleState.ard,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		background,
		out_color,
		geomState.depths,
		invdepth);

	CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHANNELS_3DGS, cudaMemcpyDeviceToDevice), debug);
	CHECK_CUDA(cudaMemcpy(imgState.pixel_invDepths, invdepth, sizeof(float) * width * height, cudaMemcpyDeviceToDevice), debug);
	return std::make_tuple(num_rendered, bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char *img_buffer,
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
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float *color_ptr = geomState.rgb;
	const int THREADS = 32;
	PerGaussianRenderCUDA<<<((B * 32) + THREADS - 1) / THREADS, THREADS>>>(
		imgState.ranges,
		binningState.point_list,
		width, height, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T,
		sampleState.ar,
		sampleState.ard,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		imgState.pixel_invDepths,
		dL_dpix,
		dL_invdepths,
		(glm::vec3 *)dL_dmean2D,
		(glm::vec4 *)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float *cov3D_ptr = geomState.cov3D;
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
		P,
		(glm::vec3 *)means3D,
		radii,
		cov3D_ptr,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		opacities,
		dL_dconic,
		dL_dopacity,
		dL_dinvdepth,
		(glm::vec3 *)dL_dmean3D,
		dL_dcov3D,
		antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDABackward<<<(P + 255) / 256, 256>>>(
		P, D, M,
		(glm::vec3 *)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		projmatrix,
		(glm::vec3 *)campos,
		(glm::vec3 *)dL_dmean2D,
		(glm::vec3 *)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3 *)dL_dscale,
		(glm::vec4 *)dL_drot,
		dL_dopacity);
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
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
	const bool antialiasing,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	// auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS_3DGS, H, W}, 0.0, float_opts);
	torch::Tensor out_invdepth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);
	std::function<char *(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

	int rendered = 0;
	int num_buckets = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		auto tup = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			sampleFunc,
			P, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			dc.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			// colors.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			// scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			// cov3D_precomp.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			// prefiltered,
			out_color.contiguous().data_ptr<float>(),
			out_invdepth.contiguous().data_ptr<float>(),
			antialiasing,
			radii.contiguous().data_ptr<int>(),
			debug);

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
	}
	return std::make_tuple(rendered, num_buckets, out_color, out_invdepth, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &opacities,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
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
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS_3DGS}, means3D.options());
	torch::Tensor dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options()); // quats {P, 3, 3}

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
											 background.contiguous().data_ptr<float>(),
											 W, H,
											 means3D.contiguous().data_ptr<float>(),
											 dc.contiguous().data_ptr<float>(),
											 sh.contiguous().data_ptr<float>(),
											 // colors.contiguous().data_ptr<float>(),
											 opacities.contiguous().data_ptr<float>(),
											 scales.data_ptr<float>(),
											 scale_modifier,
											 rotations.data_ptr<float>(),
											 // cov3D_precomp.contiguous().data_ptr<float>(),
											 viewmatrix.contiguous().data_ptr<float>(),
											 projmatrix.contiguous().data_ptr<float>(),
											 campos.contiguous().data_ptr<float>(),
											 tan_fovx,
											 tan_fovy,
											 radii.contiguous().data_ptr<int>(),
											 reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
											 reinterpret_cast<char *>(sampleBuffer.contiguous().data_ptr()),
											 dL_dout_color.contiguous().data_ptr<float>(),
											 dL_dout_invdepth.contiguous().data_ptr<float>(),
											 dL_dmeans2D.contiguous().data_ptr<float>(),
											 dL_dconic.contiguous().data_ptr<float>(),
											 dL_dopacity.contiguous().data_ptr<float>(),
											 dL_dcolors.contiguous().data_ptr<float>(),
											 dL_dinvdepths.contiguous().data_ptr<float>(),
											 dL_dmeans3D.contiguous().data_ptr<float>(),
											 dL_dcov3D.contiguous().data_ptr<float>(),
											 dL_ddc.contiguous().data_ptr<float>(),
											 dL_dsh.contiguous().data_ptr<float>(),
											 dL_dscales.contiguous().data_ptr<float>(),
											 dL_drotations.contiguous().data_ptr<float>(),
											 antialiasing,
											 debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dopacity, dL_dmeans3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}