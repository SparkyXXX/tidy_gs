# CUDA库的CMake动态编译配置

首先正常跑高斯能够使用sparse_adam，即运行train.py时可以加--optimizer_type sparse_adam参数训练，如果不行可以按照下面的操作进行配置。因为如果用不了说明没有adam.cu这个为文件，跟下面的配置步骤对不上，理论上调一调也能行，但可能还需要踩一些坑，更何况sparse_adam有2.7倍的训练加速呢，还是配一个吧

关于sparse_adam的配置如下，如果可以使用sparse_adam可以跳过

由于服务器不能进行git操作，故将代码克隆到本地，以下几步为本地操作

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive # 注意在自己电脑上运行
```

再到[glm仓库](https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903)下载压缩包，替换本地./submodules/diff-gaussian-rasterization/third_party/glm这个空目录

同样由于不能git，在本地切换分支

```bash
cd submodules/diff-gaussian-rasterization
git checkout 3dgs_accel
```

现在将本地的submodules/diff-gaussian-rasterization目录上传到服务器对应位置

以上为sparse_adam的配置

---

完成了sparse_adam的配置，以下开始为服务器上的操作

```bash
conda activate 3dgs
conda install pybind11

# 确保清除之前的安装
pip uninstall diff-gaussian-rasterization
cd /home/huangruixiang/gaussian-splatting/submodules/diff-gaussian-rasterization
rm -rf build/ dist/ *.egg-info
rm -f diff_gaussian_rasterization/*.so
```

把一些文件改成下面的样子：

./submodules/diff-gaussian-rasterization/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)
set(CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

add_library(CudaRasterizer SHARED
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.cu
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/adam.cu
    rasterize_points.cu
    conv.cu
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(CudaRasterizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endif()

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE 
    third_party/glm 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(CudaRasterizer PRIVATE 
    "${TORCH_LIBRARIES}" 
    ${Python_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

pybind11_add_module(RUIXIANG_S_NB_TOOL SHARED ext.cpp)
target_link_libraries(RUIXIANG_S_NB_TOOL PRIVATE 
    CudaRasterizer 
    "${TORCH_LIBRARIES}"
    pybind11::module
    ${TORCH_PYTHON_LIBRARY}
)
target_include_directories(RUIXIANG_S_NB_TOOL PRIVATE ${TORCH_INCLUDE_DIRS})
```

./submodules/diff-gaussian-rasterization/ext.cpp

```cpp
#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(RUIXIANG_S_NB_TOOL, m)
{
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("adamUpdate", &adamUpdate);
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}
```

./submodules/diff-gaussian-rasterization/diff-gaussian-rasterization/\_\_init\_\_.py的开头导入部分，另外改完之后查找替换把所有的\_C替换成RUIXIANG_S_NB_TOOL

```python
import os
import sys
from typing import NamedTuple
import torch.nn as nn
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import RUIXIANG_S_NB_TOOL # type: ignore
```

然后cmake编译，第一次稍微慢一些，看到警告不要紧张，包没问题的

```bash
cd ./submodules/diff-gaussian-rasterization
mkdir build
cd ./build
cmake -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") ..
cmake --build .
```

新建一个脚本文件用来一键操作，比如叫ruixiang_tool.sh，写入以下内容

```sh
#!/bin/bash

# 进入 diff-gaussian-rasterization 的 build 目录
cd submodules/diff-gaussian-rasterization/build/ || {
    echo "Error: 无法进入目录 submodules/diff-gaussian-rasterization/build/"
    exit 1
}

# 执行 cmake 构建命令
echo "开始根据你的改动进行动态构建项目..."
cmake --build . || {
    echo "Error: cmake 构建失败"
    exit 1
}

# 返回到原始目录（脚本执行前的目录）
cd ../../.. || {
    echo "Error: 无法返回上级目录"
    exit 1
}

echo "动态构建完成，让我们继续愉快地debug吧"
```

理论上已经配置好了，回到项目根目录跑一下渲染试试，没问题就说明已经成功调用了CMake编译出的.so文件了，因为这个时候其实还没有install

```bash
cd ../../..
python render.py -m ./data/Hub/output
```

随便在CUDA部分改点东西，比如在forward.cu中加下面这一句，让它每渲染一张图打印一下"Ruixiang is really NB!!!"

```cpp
if (idx == 0)
{
	printf("Ruixiang is really NB!!!\n");
}
```

然后执行

```bash
./ruixiang_tool.sh
```

就可以看到飞快的动态编译构建过程啦，debug效率$$\times$$100

![](https://cdn.jsdelivr.net/gh/SparkyXXX/Hatrix-s-Blog-Image/img/20250816022510268.png)

跑一下渲染看看改动是不是生效了

```bash
python render.py -m ./data/Hub/output
```

看到满终端的Ruixiang is really NB!!!就搞定啦

![](https://cdn.jsdelivr.net/gh/SparkyXXX/Hatrix-s-Blog-Image/img/20250816022620964.png)

然后可以愉快地删除慢的要死的./submodules/diff-gaussian-rasterization/setup.py（什么垃圾玩意儿编译一次两分钟）