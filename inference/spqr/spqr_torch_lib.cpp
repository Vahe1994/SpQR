/*
 * Copyright (C) SPQR Kernel.2024 Elvir Crncevic (elvircrn@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/python.h>
#include <torch/script.h> // One-stop header.

int spqr_matvec(
    // W and meta
    int bits, int prob_m, int prob_n,
    // Quantization
    int beta1, int beta2,
    const void *raw_data,
    void *row_offsets,
    // 32-bit
    void *col_vals,
    int nnz,
    // 16-bit
    // Input
    void *X,
    void *order,
    // Output
    void *y,
    // GPU meta
    cudaStream_t stream = nullptr,
    void *measurements = nullptr,
    uint32_t feature_flag = 0);


void spqr_mul(int64_t m,
              int64_t n,
              int64_t bits,
              int64_t beta1, int64_t beta2,
              const torch::Tensor &buff0,
              const torch::Tensor &row_offsets,
              const torch::Tensor &col_val_ptr,
              int64_t nnz,
              const torch::Tensor &X,
              torch::Tensor &Y,
              int64_t _feature_flag = 0) {
  uint32_t feature_flag = static_cast<uint32_t>(_feature_flag);
  int dev = buff0.get_device();
  // TODO: Propagate error one layer up.
  int err = spqr_matvec(
      bits, m, n, beta1, beta2, buff0.data_ptr(),
      row_offsets.data_ptr(), col_val_ptr.data_ptr(), nnz,
      X.data_ptr(), nullptr, Y.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev), nullptr, feature_flag);
}

// We need this to have a valid CMake configuration which is useful for IDE support during kernel development.
TORCH_LIBRARY(spqr_torch_lib, m) {
  m.def("spqr_mul", &spqr_mul);
}
