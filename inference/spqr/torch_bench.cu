/*
 * Copyright (C) SPQR Kernel.2024 Elvir Crncevic (elvir.crncevic@ist.ac.at)
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

#include "bit_array.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cusparse.h>

#include <vector>

namespace torch_test {

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))

using u32 = unsigned int;
using u16 = unsigned short;

__device__ __host__ CUINLINE int updiv(int x, int y) { return (x + y - 1) / y; }

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

struct _Timer {
  cudaEvent_t ce_start{}, ce_stop{};
  cudaStream_t stream;

  void start() { AT_CUDA_CHECK(cudaEventRecord(ce_start, stream)); }

  float end() {
    float time;
    AT_CUDA_CHECK(cudaEventRecord(ce_stop, stream));
    AT_CUDA_CHECK(cudaEventSynchronize(ce_stop));
    AT_CUDA_CHECK(cudaEventElapsedTime(&time, ce_start, ce_stop));
    // Returns ms
    return time;
  }

  _Timer(cudaStream_t stream) : stream(stream) {
    AT_CUDA_CHECK(cudaEventCreate(&ce_start));
    AT_CUDA_CHECK(cudaEventCreate(&ce_stop));
  }

  _Timer(_Timer &&timer) = delete;
  _Timer(const _Timer &timer) = delete;

  ~_Timer() {
    AT_CUDA_CHECK(cudaEventDestroy(ce_start));
    AT_CUDA_CHECK(cudaEventDestroy(ce_stop));
  }
};

int torch_matvec(int m, int n, void *dequantized_w, void *X, void *y,
                 void *measurements, cudaStream_t stream) {
  if (m == 0 || n == 0) {
    return 0;
  }
  torch::Tensor deq_w_tensor = torch::from_blob(
      dequantized_w, {m * n},
      torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));

  auto deq_reshaped = torch::reshape(deq_w_tensor, {m, n}).contiguous();

  torch::Tensor X_tensor = torch::from_blob(
      X, {n}, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));

  torch::Tensor result_tensor = torch::from_blob(
      y, {m}, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));

  _Timer *timer;
  if (measurements) {
    timer = new _Timer(stream);
    timer->start();
  }

  // Make sure that the compiler doesn't optimize this away
  torch::mv_out(result_tensor, deq_reshaped, X_tensor);
  (void)result_tensor;

  CHECK_CUDA(cudaDeviceSynchronize());

  if (measurements) {
    ((float *)measurements)[0] = timer->end();
    delete timer;
  }
  return 0;
}
} // namespace torch_test
