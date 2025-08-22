#include <torch/extension.h>
#include "auxiliary.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include "cuda_rasterizer/adam.cuh"

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const bool antialiasing)
{
	const int P = means3D.size(0);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS_3DGS, image_height, image_width}, 0.0, float_opts);
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
		int M = (sh.size(0) != 0) ? sh.size(1) : 0;
		auto tup = rasterizeForward(
			geomFunc,
			binningFunc,
			imgFunc,
			sampleFunc,
			P,
			means3D.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			degree,
			M,
			dc.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			background.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			image_width,
			image_height,
			tan_fovx,
			tan_fovy,
			out_color.contiguous().data_ptr<float>(),
			antialiasing,
			radii.contiguous().data_ptr<int>());

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
	}
	return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
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
	// const torch::Tensor &dL_dout_invdepth,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const int B,
	const torch::Tensor &sampleBuffer,
	const bool antialiasing)
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
		rasterizeBackward(P, degree, M, R, B,
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
											 // dL_dout_invdepth.contiguous().data_ptr<float>(),
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
											 antialiasing);
	}

	return std::make_tuple(dL_dmeans2D, dL_dopacity, dL_dmeans3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
}

PYBIND11_MODULE(RUIXIANG_S_NB_TOOL, m)
{
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
    m.def("adamUpdate", &adamUpdate);
}