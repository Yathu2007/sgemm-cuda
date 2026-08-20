cmake_minimum_required(VERSION 3.20)
project(sgemm-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# RTX 4050 Laptop / Ada AD107 is compute capability 8.9. Confirmed by device_query below
set(CMAKE_CUDA_ARCHITECTURES 89)

# Dumps the hardware numbers the blocktiling parameters will be chosen against.
add_executable(device_query tools/device_query.cu)
