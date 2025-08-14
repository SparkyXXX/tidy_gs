#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize_gaussians_forward", &RasterizeGaussiansForwardCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("adamUpdate", &adamUpdate);
  m.def("markVisible", &markVisible);
  m.def("fusedssim_forward", &fusedssim_forward);
  m.def("fusedssim_backward", &fusedssim_backward);
}