#pragma once

#include <cstdio>
#include <torch/extension.h>
#define GLM_FORCE_CUDA
#define CUDA_VERSION 11080
#include <glm/glm.hpp>

#define NUM_CHANNELS_3DGS 3
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)
#define FRUSTUM_THERSOLD 0.2f
#define COV2D_COEF 1.3f
#define H_VAR 0.3f

#define CHECK_CUDA(A, debug)                                                                                           \
	A;                                                                                                                 \
	if (debug)                                                                                                         \
	{                                                                                                                  \
		auto ret = cudaDeviceSynchronize();                                                                            \
		if (ret != cudaSuccess)                                                                                        \
		{                                                                                                              \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
			throw std::runtime_error(cudaGetErrorString(ret));                                                         \
		}                                                                                                              \
	}

__device__ constexpr float SH_C0 = 0.28209479177387814f;
__device__ constexpr float SH_C1 = 0.4886025119029199f;
__device__ constexpr float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f};
__device__ constexpr float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f};

__device__ __forceinline__ float ndc2Pix(const float ndc, const int pix)
{
	return ((ndc + 1.0f) * static_cast<float>(pix) - 1.0f) * 0.5f;
}

__device__ __forceinline__ void getRect(const glm::vec2 p, const int radius,
	uint2 *rect_min, uint2 *rect_max, const dim3 grid)
{
	const glm::vec2 r{static_cast<float>(radius), static_cast<float>(radius)};
	rect_min->x = min(grid.x, max(0, static_cast<int>((p.x - r.x) / BLOCK_X)));
	rect_min->y = min(grid.y, max(0, static_cast<int>((p.y - r.y) / BLOCK_Y)));
	rect_max->x = min(grid.x, max(0, static_cast<int>((p.x + r.x + BLOCK_X - 1) / BLOCK_X)));
	rect_max->y = min(grid.y, max(0, static_cast<int>((p.y + r.y + BLOCK_Y - 1) / BLOCK_Y)));
}

// vec3 -> vec4, w=1 表示点（会应用平移分量）
__device__ __forceinline__ glm::vec3 transformPoint4x3(const glm::vec3 p, const float *matrix)
{
	const glm::vec3 transformed =
		{
			matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
			matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
			matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		};
	return transformed;
}

// 返回完整 vec4（包括 w 分量）
__device__ __forceinline__ glm::vec4 transformPoint4x4(const glm::vec3 p, const float *matrix)
{
	const glm::vec4 transformed =
		{
			matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
			matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
			matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
			matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
	return transformed;
}

// 纯向量变换（忽略平移分量，w=0）
__device__ __forceinline__ glm::vec3 transformVec4x3Transpose(const glm::vec3 p, const float *matrix)
{
	const glm::vec3 transformed =
		{
			matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
			matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
			matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
		};
	return transformed;
}

struct GeometryState
{
	size_t scan_size;
	float *depths;
	char *scanning_space;
	bool *clamped;
	int *internal_radii;
	glm::vec2 *means2D;
	float *cov3D;
	glm::vec4 *conic_opacity;
	float *rgb;
	uint32_t *point_offsets;
	uint32_t *tiles_touched;

	static GeometryState fromChunk(char *&chunk, size_t P);
};

struct ImageState
{
	uint32_t *bucket_count;
	uint32_t *bucket_offsets;
	size_t bucket_count_scan_size;
	char *bucket_count_scanning_space;
	float *pixel_colors;
	float *pixel_invDepths;
	uint32_t *max_contrib;

	size_t scan_size;
	uint2 *ranges;
	uint32_t *n_contrib;
	float *accum_alpha;
	char *contrib_scan;

	static ImageState fromChunk(char *&chunk, size_t N);
};

struct BinningState
{
	size_t sorting_size;
	uint64_t *point_list_keys_unsorted;
	uint64_t *point_list_keys;
	uint32_t *point_list_unsorted;
	uint32_t *point_list;
	char *list_sorting_space;

	static BinningState fromChunk(char *&chunk, size_t P);
};

struct SampleState
{
	uint32_t *bucket_to_tile;
	float *T;
	float *ar;
	float *ard;

	static SampleState fromChunk(char *&chunk, size_t C);
};

template <typename T>
size_t required(size_t P)
{
	char *size = nullptr;
	T::fromChunk(size, P);
	return reinterpret_cast<size_t>(size) + 128;
}

template <typename T>
void obtain(char *&chunk, T *&ptr, std::size_t count, std::size_t alignment)
{
	std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T *>(offset);
	chunk = reinterpret_cast<char *>(ptr + count);
}

uint32_t get_higher_msb(uint32_t n);
__global__ void duplicate_with_keys(
	int num_points,
	const glm::vec2 *positions2d,
	const glm::vec4 *__restrict__ conic_opacity,
	const uint32_t *offsets,
	float *depths,
	int *radii,
	dim3 grid,
	uint64_t *gaussian_keys_unsorted,
	uint32_t *gaussian_values_unsorted);
__global__ void identify_tile_ranges(int len_point_list, uint64_t *point_list_keys, uint2 *tile_id_ranges);
__global__ void per_tile_bucket_count(int num_tiles, uint2 *tile_id_ranges, uint32_t *bucket_count);
std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t);