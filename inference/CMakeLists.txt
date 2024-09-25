cmake_minimum_required(VERSION 3.25)
project(spqr CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.10 COMPONENTS Interpreter Development)
#find_package(pybind11 REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)

# Find Torch
find_package(Torch REQUIRED PATHS ${TORCH_PATH})

add_executable(spqr
        spqr/spqr_cuda.cpp
        spqr/spqr_cuda_kernel.cu
        spqr/torch_bench.cu
)

add_library(spqr_lib SHARED
        spqr/spqr_torch_lib.cpp
        spqr/spqr_cuda_kernel.cu
        spqr/torch_bench.cu
)


target_compile_options(spqr_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_include_directories(spqr_lib PUBLIC spqr cnpy ${TORCH_INCLUDE_DIRS})

target_compile_definitions(spqr PUBLIC PYBIND_SKIP=1)
target_compile_options(spqr PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_include_directories(spqr PUBLIC spqr cnpy ${TORCH_INCLUDE_DIRS})


target_link_libraries(spqr PUBLIC ${TORCH_LIBRARIES} Python::Python)
target_link_libraries(spqr_lib PUBLIC ${TORCH_LIBRARIES} Python::Python)