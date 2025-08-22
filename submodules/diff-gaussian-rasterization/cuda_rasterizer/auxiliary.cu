#include "auxiliary.cuh"
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#define CUDA_VERSION 11080
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace cg = cooperative_groups;

uint32_t get_higher_msb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
        {
            msb += step;
        }
        else
        {
            msb -= step;
        }
    }
    if (n >> msb)
    {
        msb++;
    }
    return msb;
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const glm::vec4 co)
{
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template <uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
    const glm::vec4 co,
    const glm::vec2 mean,
    const glm::vec2 rect_min,
    const glm::vec2 rect_max,
    glm::vec2 *max_contribution_position)
{
    const float x_min_diff = rect_min.x - mean.x;
    const float x_is_left = x_min_diff > 0.0f;
    const float x_not_in_range = x_is_left + (mean.x > rect_max.x);
    const float y_min_diff = rect_min.y - mean.y;
    const float y_is_above = y_min_diff > 0.0f;
    const float y_not_in_range = y_is_above + (mean.y > rect_max.y);

    *max_contribution_position = mean;
    float max_contribution_value = 0.0f;
    if (x_not_in_range + y_not_in_range > 0.0f)
    {
        const float x_edge = x_is_left * rect_min.x + (1.0f - x_is_left) * rect_max.x;
        const float y_edge = y_is_above * rect_min.y + (1.0f - y_is_above) * rect_max.y;
        const float x_step = copysign(static_cast<float>(PATCH_WIDTH), x_min_diff);
        const float y_step = copysign(static_cast<float>(PATCH_HEIGHT), y_min_diff);
        const float x_diff_to_edge = mean.x - x_edge;
        const float y_diff_to_edge = mean.y - y_edge;
        const float x_inv = __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x);
        const float y_inv = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z);
    	// hrx: why swap x_not_in_range and y_not_in_range???
        const float x_diff_normalized = y_not_in_range * __saturatef((x_step * co.x * x_diff_to_edge + x_step * co.y * y_diff_to_edge) * x_inv);
        const float y_diff_normalized = x_not_in_range * __saturatef((y_step * co.y * x_diff_to_edge + y_step * co.z * y_diff_to_edge) * y_inv);

        *max_contribution_position = {x_edge + x_diff_normalized * x_step, y_edge + y_diff_normalized * y_step};
        glm::vec2 max_contribution_distance = mean - *max_contribution_position;
        max_contribution_value = evaluate_opacity_factor(max_contribution_distance.x, max_contribution_distance.y, co);
    }
    return max_contribution_value;
}

__global__ void duplicate_with_keys(
    int num_points,
    const glm::vec2 *positions2d,
    const glm::vec4 *__restrict__ conic_opacity,
    const uint32_t *offsets,
    float *depths,
    int *radii,
    dim3 grid,
    uint64_t *gaussian_keys_unsorted,
    uint32_t *gaussian_values_unsorted)
{
    const auto idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
    {
        return;
    }
    if (radii[idx] > 0)
    {
        uint32_t offset_from_in_pointlist = (idx == 0) ? 0 : offsets[idx - 1];
        const uint32_t offset_to_in_pointlist = offsets[idx];
        const glm::vec2 position2d = positions2d[idx];
        const glm::vec4 conic = conic_opacity[idx];

        uint2 rect_min, rect_max;
    	getRect(position2d, radii[idx], &rect_min, &rect_max, grid);

        const float opacity_factor_thersold = logf(conic.w / (1.0f / 255.0f));
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
                const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);
                glm::vec2 max_contribution_position;
                float max_opacity_factor = max_contrib_power_rect_gaussian_float<BLOCK_X - 1, BLOCK_Y - 1>
                    (conic, position2d, tile_min, tile_max, &max_contribution_position);

                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *reinterpret_cast<uint32_t *>(&depths[idx]);

                if (max_opacity_factor <= opacity_factor_thersold)
                {
                    gaussian_keys_unsorted[offset_from_in_pointlist] = key;
                    gaussian_values_unsorted[offset_from_in_pointlist] = idx;
					offset_from_in_pointlist++;
                }
            }
        }

        for (; offset_from_in_pointlist < offset_to_in_pointlist; offset_from_in_pointlist++)
        {
        	uint64_t key = static_cast<uint32_t>(-1);
            key <<= 32;
            float depth = FLT_MAX;
            key |= *reinterpret_cast<uint32_t *>(&depth);
            gaussian_values_unsorted[offset_from_in_pointlist] = 0xFFFFFFFFu;
            gaussian_keys_unsorted[offset_from_in_pointlist] = key;
        }
    }
}

__global__ void identify_tile_ranges(int len_point_list, uint64_t *point_list_keys, uint2 *tile_id_ranges)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= len_point_list)
	{
		return;
	}
	const uint64_t key = point_list_keys[idx];
	const uint32_t current_tile = key >> 32;
	bool valid_tile = (current_tile != 0xFFFFFFFFu);

	if (idx == 0)
	{
		tile_id_ranges[current_tile].x = 0;
	}
	else
	{
		uint32_t previous_tile = point_list_keys[idx - 1] >> 32;
		if (current_tile != previous_tile)
		{
			tile_id_ranges[previous_tile].y = idx;
			if (valid_tile)
			{
				tile_id_ranges[current_tile].x = idx;
			}
		}
	}
	if ((idx == len_point_list - 1) && (valid_tile))
	{
		tile_id_ranges[current_tile].y = len_point_list;
	}
}

__global__ void per_tile_bucket_count(int num_tiles, uint2 *tile_id_ranges, uint32_t *bucket_count)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= num_tiles)
	{
		return;
	}
	const uint2 tile_id_range = tile_id_ranges[idx];
	const int num_splats = tile_id_range.y - tile_id_range.x;
	const int num_buckets = (num_splats + 31) / 32;
	bucket_count[idx] = static_cast<uint32_t>(num_buckets);
}

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({static_cast<long long>(N)});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

GeometryState GeometryState::fromChunk(char *&chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

ImageState ImageState::fromChunk(char *&chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	int *dummy = nullptr;
	int *wummy = nullptr;
	cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
	obtain(chunk, img.contrib_scan, img.scan_size, 128);
	obtain(chunk, img.max_contrib, N, 128);
	obtain(chunk, img.pixel_colors, N * NUM_CHANNELS_3DGS, 128);
	obtain(chunk, img.pixel_invDepths, N, 128);
	obtain(chunk, img.bucket_count, N, 128);
	obtain(chunk, img.bucket_offsets, N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
	obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);
	return img;
}

SampleState SampleState::fromChunk(char *&chunk, size_t C)
{
	SampleState sample;
	obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ar, NUM_CHANNELS_3DGS * C * BLOCK_SIZE, 128);
	obtain(chunk, sample.ard, C * BLOCK_SIZE, 128);
	return sample;
}

BinningState BinningState::fromChunk(char *&chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}