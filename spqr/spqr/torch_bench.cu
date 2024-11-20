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

#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include <cusparse.h>

namespace torch_test {

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))

using u32 = unsigned int;
using u16 = unsigned short;

__device__ __host__ CUINLINE int updiv(int x, int y) { return (x + y - 1) / y; }

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

  Timer *timer;
  if (measurements) {
    timer = new Timer(stream);
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
