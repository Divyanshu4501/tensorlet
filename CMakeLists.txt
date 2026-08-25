cmake_minimum_required(VERSION 3.18)

project(tensorlet_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(tensorlet_cuda
    csrc/bindings.cpp
    csrc/ops.cu
)

target_compile_options(tensorlet_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)