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