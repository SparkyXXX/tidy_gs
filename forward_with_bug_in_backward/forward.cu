#include "forward.cuh"
#include "auxiliary.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace cg = cooperative_groups;

__device__ glm::vec3 compute_color_from_sh(const int degree, const glm::vec3 position3d_w, const glm::vec3 camera_pos,
										   const glm::vec3 dc, const glm::vec3 *sh)
{
	glm::vec3 direction = position3d_w - camera_pos;
	direction = direction / glm::length(direction);

	glm::vec3 result = SH_C0 * dc;
	if (degree > 0)
	{
		const float x = direction.x;
		const float y = direction.y;
		const float z = direction.z;
		result = result - SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];

		if (degree > 1)
		{
			const float xx = x * x, yy = y * y, zz = z * z;
			const float xy = x * y, yz = y * z, xz = x * z;
			result = result +
					 SH_C2[0] * xy * sh[3] +
					 SH_C2[1] * yz * sh[4] +
					 SH_C2[2] * (2.0f * zz - xx - yy) * sh[5] +
					 SH_C2[3] * xz * sh[6] +
					 SH_C2[4] * (xx - yy) * sh[7];

			if (degree > 2)
			{
				result = result +
						 SH_C3[0] * y * (3.0f * xx - yy) * sh[8] +
						 SH_C3[1] * xy * z * sh[9] +
						 SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[10] +
						 SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[11] +
						 SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[12] +
						 SH_C3[5] * z * (xx - yy) * sh[13] +
						 SH_C3[6] * x * (xx - 3.0f * yy) * sh[14];
			}
		}
	}
	result += 0.5f;
	return glm::max(result, 0.0f);
}

__device__ void compute_cov3d(const glm::vec3 scale, const glm::vec4 rotation, float *cov3d)
{
	auto S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	const float r = rotation.x;
	const float x = rotation.y;
	const float y = rotation.z;
	const float z = rotation.w;
	const auto R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	const auto M = S * R;
	glm::mat3 sigma = glm::transpose(M) * M;
	cov3d[0] = sigma[0][0];
	cov3d[1] = sigma[0][1];
	cov3d[2] = sigma[0][2];
	cov3d[3] = sigma[1][1];
	cov3d[4] = sigma[1][2];
	cov3d[5] = sigma[2][2];
}

__device__ glm::vec3 compute_cov2d(const float *cov3d, const glm::vec3 position3d_w, const float *view_matrix,
								   const float focal_x, const float focal_y, const float tan_fovx, const float tan_fovy)
{
	const auto position3d_c = transformPoint4x3(position3d_w, view_matrix);
	float x = position3d_c.x;
	float y = position3d_c.y;
	float z = position3d_c.z;
	// hrx: 透视除法，裁剪限幅，没理解
	const float limx = COV2D_COEF * tan_fovx;
	const float limy = COV2D_COEF * tan_fovy;
	const float txtz = x / z;
	const float tytz = y / z;
	x = min(limx, max(-limx, txtz)) * z;
	y = min(limy, max(-limy, tytz)) * z;

	const auto J = glm::mat3(
		focal_x / z, 0.0f, -(focal_x * x) / (z * z),
		0.0f, focal_y / z, -(focal_y * y) / (z * z),
		0, 0, 0);
	const auto W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);
	const glm::mat3 T = W * J;
	const auto vrk = glm::mat3(
		cov3d[0], cov3d[1], cov3d[2],
		cov3d[1], cov3d[3], cov3d[4],
		cov3d[2], cov3d[4], cov3d[5]);
	glm::mat3 cov2d_temp = glm::transpose(T) * vrk * T;
	return glm::vec3{cov2d_temp[0][0], cov2d_temp[0][1], cov2d_temp[1][1]};
}

__global__ void preprocessForward(
	int num_points,
	const float *positions3d_world,
	const glm::vec3 *scales,
	const glm::vec4 *rotations,
	int degree,
	int max_coef,
	const float *dcs,
	const float *shs,
	const float *opacities,

	const float *view_matrix,
	const float *projection_matrix,
	const glm::vec3 *camera_poses,
	int width,
	int height,
	float focal_x,
	float focal_y,
	float tan_fovx,
	float tan_fovy,

	bool *clamped,
	int *radii,
	glm::vec2 *points_xy_image,
	float *depths,
	float *cov3ds,
	float *rgb,
	glm::vec4 *conic_opacities,
	dim3 grid,
	uint32_t *tiles_touched,
	bool antialiasing)
{
	const auto idx = cg::this_grid().thread_rank();
	if (idx >= num_points)
	{
		return;
	}
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	const glm::vec3 position3d_w = {positions3d_world[3 * idx], positions3d_world[3 * idx + 1], positions3d_world[3 * idx + 2]};
	glm::vec3 position3d_c = transformPoint4x3(position3d_w, view_matrix);
	depths[idx] = position3d_c.z;
	if (position3d_c.z <= FRUSTUM_THERSOLD)
	{
		return;
	}

	glm::vec4 position3d_ndc_temp = transformPoint4x4(position3d_w, projection_matrix);
	float w_scale = 1.0f / (position3d_ndc_temp.w + 0.0000001f);
	glm::vec3 position_ndc = {position3d_ndc_temp.x * w_scale, position3d_ndc_temp.y * w_scale, position3d_ndc_temp.z * w_scale};

	float *cov3d = cov3ds + idx * 6;
	compute_cov3d(scales[idx], rotations[idx], cov3d);
	glm::vec3 cov2d = compute_cov2d(cov3d, position3d_w, view_matrix, focal_x, focal_y, tan_fovx, tan_fovy);

	const float det_cov2d = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
	cov2d.x += H_VAR;
	cov2d.z += H_VAR;
	const float det_cov2d_plus = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
	float cov2d_gain = 1.0f;
	if (antialiasing)
	{
		cov2d_gain = sqrt(max(0.000025f, det_cov2d / det_cov2d_plus));
	}

	// const float det = det_cov2d_plus;
	if (det_cov2d_plus == 0.0f)
	{
		return;
	}
	float det_cov2d_plus_inv = 1.f / det_cov2d_plus;
	glm::vec3 cov2d_inv = {cov2d.z * det_cov2d_plus_inv, -cov2d.y * det_cov2d_plus_inv, cov2d.x * det_cov2d_plus_inv};

	float mid = 0.5f * (cov2d.x + cov2d.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det_cov2d_plus));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det_cov2d_plus));
	float gs_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	glm::vec2 position2d = {ndc2Pix(position_ndc.x, width), ndc2Pix(position_ndc.y, height)};
	uint2 rect_min, rect_max;
	getRect(position2d, gs_radius, &rect_min, &rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
	{
		return;
	}
	const glm::vec3 dc_vec = reinterpret_cast<const glm::vec3 *>(dcs)[idx];
	const glm::vec3 *sh_vec = reinterpret_cast<const glm::vec3 *>(shs) + idx * max_coef;
	glm::vec3 result = compute_color_from_sh(degree, position3d_w, *camera_poses, dc_vec, sh_vec);
	rgb[idx * NUM_CHANNELS_3DGS + 0] = result.x;
	rgb[idx * NUM_CHANNELS_3DGS + 1] = result.y;
	rgb[idx * NUM_CHANNELS_3DGS + 2] = result.z;
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);

	radii[idx] = gs_radius;
	points_xy_image[idx] = position2d;
	float opacity = opacities[idx];
	conic_opacities[idx] = {cov2d_inv.x, cov2d_inv.y, cov2d_inv.z, opacity * cov2d_gain};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__global__ void __launch_bounds__(BLOCK_SIZE) renderForward(
	int width,
	int height,
	const glm::vec2 *__restrict__ position2d,
	const float *__restrict__ features,
	const float *__restrict__ bg_color,
	const glm::vec4 *__restrict__ conic_opacities,

	const uint2 *__restrict__ point_ranges_in_point_list_for_current_tile,
	const uint32_t *__restrict__ point_list,
	const uint32_t *__restrict__ bucket_offset_in_a_tile_in_global_bucket_array,
	uint32_t *__restrict__ bucket_in_which_tile,

	uint32_t *__restrict__ n_contributor,
	uint32_t *__restrict__ max_contributor,
	float *__restrict__ out_color,
	float *__restrict__ final_T)
{
	const auto block = cg::this_thread_block();
	const uint32_t num_horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_left_up = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_X};
	// const uint2 pix_right_down = {min(pix_left_up.x + BLOCK_X, width), min(pix_left_up.y + BLOCK_Y, height)};
	const uint2 current_pix = {pix_left_up.x + block.thread_index().x, pix_left_up.y + block.thread_index().y};
	const uint32_t current_pix_id = current_pix.y * width + current_pix.x;
	const uint32_t tile_id = block.group_index().y * num_horizontal_blocks + block.group_index().x;
	const uint32_t tile_offset_in_bucket_global_array = (tile_id == 0) ? 0 : bucket_offset_in_a_tile_in_global_bucket_array[tile_id - 1];
	const uint2 point_range_in_point_list_for_current_tile = point_ranges_in_point_list_for_current_tile[tile_id];

	int todo = point_range_in_point_list_for_current_tile.y - point_range_in_point_list_for_current_tile.x;
	const int num_buckets = (todo + 31) / 32;
	for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; i++)
	{
		int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
		if (bucket_idx < num_buckets)
		{
			bucket_in_which_tile[tile_offset_in_bucket_global_array + bucket_idx] = tile_id;
		}
	}

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ glm::vec2 collected_xy[BLOCK_SIZE];
	__shared__ glm::vec4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[NUM_CHANNELS_3DGS] = {0};

	const int rounds = (todo + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const bool inside = (current_pix.x < width) && (current_pix.y < height);
	bool done = !inside;
	for (int i = 0; i < rounds; i++, todo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
		{
			break;
		}
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (point_range_in_point_list_for_current_tile.x + progress < point_range_in_point_list_for_current_tile.y)
		{
			int coll_id = point_list[point_range_in_point_list_for_current_tile.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = position2d[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacities[coll_id];
		}
		block.sync();

		for (int j = 0; (!done) && (j < min(BLOCK_SIZE, todo)); j++)
		{
			contributor++;
			glm::vec2 xy = collected_xy[j];
			glm::vec2 pixf = {static_cast<float>(current_pix.x), static_cast<float>(current_pix.y)};
			glm::vec2 distance = xy - pixf;
			glm::vec4 con_opacity = collected_conic_opacity[j];
			float power = -0.5f * (con_opacity.x * distance.x * distance.x + con_opacity.z * distance.y * distance.y) - con_opacity.y * distance.x * distance.y;
			if (power > 0.0f)
			{
				continue;
			}

			float alpha = min(0.99f, con_opacity.w * exp(power));
			if (alpha < 1.0f / 255.0f)
			{
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 1e-4f)
			{
				done = true;
				continue;
			}

			for (int ch = 0; ch < NUM_CHANNELS_3DGS; ch++)
			{
				C[ch] += features[collected_id[j] * NUM_CHANNELS_3DGS + ch] * alpha * T;
			}
			T = test_T;
			last_contributor = contributor;
		}
	}
	if (inside)
	{
		final_T[current_pix_id] = T;
		n_contributor[current_pix_id] = last_contributor;
		for (int ch = 0; ch < NUM_CHANNELS_3DGS; ch++)
		{
			out_color[ch * height * width + current_pix_id] = C[ch] + T * bg_color[ch];
		}
	}

	typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
	__shared__ BlockReduce::TempStorage temp_storage;
	last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
	if (block.thread_rank() == 0)
	{
		max_contributor[tile_id] = last_contributor;
	}
}

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
	int *radii)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(num_points);
	char *chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, num_points);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	preprocessForward<<<(num_points + 255) / 256, 256>>>(
		num_points,
		positions3d,
		(glm::vec3 *)scales,
		(glm::vec4 *)rotations,
		degree,
		max_coef,
		dcs,
		shs,
		opacities,
		view_matrix,
		projection_matrix,
		(glm::vec3*)(camera_poses),
		width,
		height,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,

		geomState.clamped,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		antialiasing);

	cub::DeviceScan::InclusiveSum(
		geomState.scanning_space,
		geomState.scan_size,
		geomState.tiles_touched,
		geomState.point_offsets,
		num_points);

	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char *binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicate_with_keys<<<(num_points + 255) / 256, 256>>>(
		num_points,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.point_offsets,
		geomState.depths,
		radii,
		tile_grid,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted);
	int bit = get_higher_msb(tile_grid.x * tile_grid.y);

	cub::DeviceRadixSort::SortPairs(
				   binningState.list_sorting_space,
				   binningState.sorting_size,
				   binningState.point_list_keys_unsorted, binningState.point_list_keys,
				   binningState.point_list_unsorted, binningState.point_list,
				   num_rendered, 0, 32 + bit);
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	if (num_rendered > 0)
	{
		identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	}

	int num_tiles = tile_grid.x * tile_grid.y;
	per_tile_bucket_count<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);
	cub::DeviceScan::InclusiveSum(
		imgState.bucket_count_scanning_space,
		imgState.bucket_count_scan_size,
		imgState.bucket_count,
		imgState.bucket_offsets,
		num_tiles);
	unsigned int bucket_sum;

	cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	size_t sample_chunk_size = required<SampleState>(bucket_sum);
	char *sample_chunkptr = sampleBuffer(sample_chunk_size);
	SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);

	const float *feature_ptr = geomState.rgb;
	renderForward<<<tile_grid, block>>>(
		width,
		height,
		geomState.means2D,
		feature_ptr,
		background,
		geomState.conic_opacity,
		imgState.ranges,
		binningState.point_list,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		imgState.n_contrib,
		imgState.max_contrib,
		out_color,
		imgState.accum_alpha);
	cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHANNELS_3DGS, cudaMemcpyDeviceToDevice);
	return std::make_tuple(num_rendered, bucket_sum);
}