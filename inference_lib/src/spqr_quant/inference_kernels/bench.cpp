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

#include <cfloat>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

int spqr_matvec(
    // W and meta
    int bits, int prob_m, int prob_n,
    // Quantization
    int beta1, int beta2, const void *raw_in_order, const void *raw_dense_data,
    // 32-bit
    int row_offsets_len, void *row_offsets,
    // 32-bit
    void *col_vals, int nnz,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y,
    // GPU meta
    cudaStream_t stream = nullptr, void *measurements = nullptr,
    uint32_t feature_flag = 0);

struct QuantizedLinear {
  int m, n, row_offsets_count, nnz;
  uint64_t *d_dense_weights;
  uint32_t *d_col_vals;
  uint32_t *d_row_offsets;

  void free() {
    cudaFree(d_dense_weights);
    cudaFree(d_col_vals);
    cudaFree(d_row_offsets);
  }
};

template <class T> T *device_from_size(int s) {
  T *d_buff;
  cudaMalloc(reinterpret_cast<void **>(&d_buff), sizeof(T) * s);
  return d_buff;
}

template <class T>
T *device_from_file(const std::string &file_path) {
  // Open the binary file
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + file_path);
  }

  // Get the file size
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Allocate a buffer and read the data
  std::vector<T> buffer(size / sizeof(T));
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Error reading file: " + file_path);
  }

  T *d_buff;
  cudaMalloc(reinterpret_cast<void **>(&d_buff), sizeof(T) * buffer.size());

  cudaDeviceSynchronize();

  cudaMemcpy(d_buff, buffer.data(), sizeof(T) * buffer.size(),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  return d_buff;
}

using XType = uint16_t;

struct Result {
  float min;
  float mean;
};

Result mul_with_time(const QuantizedLinear &d_q, XType *d_x, XType *d_y,
                     float *measurements, int times) {
  float mmin = FLT_MAX;
  float mmean = 0;
  for (int i = 0; i < times; i++) {
    spqr_matvec(3, d_q.m, d_q.n, 16, 16, nullptr, d_q.d_dense_weights,
                d_q.row_offsets_count, d_q.d_row_offsets, d_q.d_col_vals,
                d_q.nnz, d_x, d_y, nullptr, measurements + i);
    mmin = std::min(mmin, measurements[i]);
    mmean += measurements[i];
  }
  return Result{.min = mmin, .mean = mmean / times};
}

QuantizedLinear from_path(const std::string &base_path) {
  std::ifstream meta_stream(base_path + "meta.txt");
  int m, n, row_offsets;
  meta_stream >> m >> n >> row_offsets;

  return QuantizedLinear{
      .m = m,
      .n = n,
      .row_offsets_count = row_offsets,
      .d_dense_weights =
          device_from_file<uint64_t>(base_path + "dense_weight.bin"),
      .d_col_vals = device_from_file<uint32_t>(base_path + "col_vals.bin"),
      .d_row_offsets =
          device_from_file<uint32_t>(base_path + "row_offsets.bin")};
}

int main() {
  std::string tag = "baseline_csr_v3";
  std::ofstream results("results.txt", std::ios_base::app);
  static constexpr int XY_SIZE = 11008 * 3;
  static constexpr int NUM_REPS = 512;
  int num_layers = 20;
  auto d_x = device_from_size<uint16_t>(XY_SIZE);
  auto d_y = device_from_size<uint16_t>(XY_SIZE);
  const std::vector<std::string> &layer_names{
      "mlp.down_proj",    "mlp.gate_proj",    "mlp.up_proj",
      "self_attn.k_proj", "self_attn.o_proj", "self_attn.q_proj",
      "self_attn.v_proj"};

  auto measurements = new float[NUM_REPS];

  float mean_runtime = 0.f;
  int tests{};

  for (int i = 0; i < num_layers; i++) {
    for (const auto &layer_name : layer_names) {
      std::string quant_linear_path =
          "/home/elvircrn/CLionProjects/spqr_kernel/data/"
          "output_identity_compressed_libtorch/" +
          std::to_string(i) + "/" + layer_name + "/";

      std::string quant_linear_path_ptcsr =
          "/home/elvircrn/CLionProjects/spqr_kernel/data/"
          "output_identity_compressed_ptcsr_libtorch/" +
          std::to_string(i) + "/" + layer_name + "/";

      QuantizedLinear quantized_linear = from_path(quant_linear_path);
      auto result =
          mul_with_time(quantized_linear, d_x, d_y, measurements, NUM_REPS);

      QuantizedLinear quantized_linear_ptcsr =
          from_path(quant_linear_path_ptcsr);
      auto result_ptcsr = mul_with_time(quantized_linear_ptcsr, d_x, d_y,
                                        measurements, NUM_REPS);

      mean_runtime += std::min(result_ptcsr.min, result.min);

      std::cout << std::left << std::setw(3) << i << " " << std::left
                << std::setw(20) << layer_name << "     " << std::setw(5)
                << std::left << std::setprecision(3)
                << std::min(result.min, result_ptcsr.min) << std::endl;
      tests++;

      quantized_linear.free();
      quantized_linear_ptcsr.free();
    }
  }

  results << std::left << std::setw(16) << tag << " " << (mean_runtime / tests)
          << std::endl;

  delete[] measurements;

  return 0;
}
