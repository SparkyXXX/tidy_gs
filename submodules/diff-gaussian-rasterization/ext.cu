#include <torch/extension.h>
#include "cuda_rasterizer/rasterizer_impl.cuh"
#include "cuda_rasterizer/adam.cuh"

PYBIND11_MODULE(RUIXIANG_S_NB_TOOL, m)
{
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
    m.def("adamUpdate", &adamUpdate);
}