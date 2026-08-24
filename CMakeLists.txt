cmake_minimum_required(VERSION 3.20)
project(sgemm-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# RTX 4050 Laptop / Ada AD107 is compute capability 8.9. Confirmed by
# device_query below rather than assumed.
set(CMAKE_CUDA_ARCHITECTURES 89)

# -O3 for speed, -lineinfo for Nsight Compute source correlation, and
# --ptxas-options=-v so register counts land in the build log (later rungs need
# them to reason about occupancy).
# Deliberately NO --use_fast_math: it changes numerics and would make the
# cuBLAS comparison dishonest.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
                    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm
  src/main.cu
  src/runner.cu
  src/utils.cu
)
target_include_directories(sgemm PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
target_link_libraries(sgemm PRIVATE CUDA::cublas)

# Dumps the hardware numbers the blocktiling parameters will be chosen against.
add_executable(device_query tools/device_query.cu)

# Reproduces the split-k investigation behind the exactly-zero relative error at
# N >= 2048: compares cuBLAS and a sequential-k kernel against a float64 host
# reference. OpenMP only shortens that host loop - the tool builds without it
# and reports identical numbers.
add_executable(accuracy_probe tools/accuracy_probe.cu)
target_link_libraries(accuracy_probe PRIVATE CUDA::cublas)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_compile_options(accuracy_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  target_link_libraries(accuracy_probe PRIVATE OpenMP::OpenMP_CXX)
endif()
