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
#include <c10/util/Exception.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/python.h>
#include <vector>
#include <torch/script.h> // One-stop header.

int spqr_matvec(
    // W and meta
    int bits,
    int prob_m,
    int prob_n,
    // Quantization
    int beta1,
    int beta2,
    const void *raw_in_order,
    const void *raw_dense_data,
    // 32-bit
    int row_offsets_len,
    void *row_offsets,
    // 32-bit
    void *col_vals,
    int nnz,
    // 16-bit
    // Input
    void *X,
    // Output
    void *y,
    // GPU meta
    cudaStream_t stream = nullptr,
    void *measurements = nullptr,
    uint32_t feature_flag = 0);

void spqr_mul(int64_t m,
                 int64_t n,
                 int64_t bits,
                 int64_t beta1,
                 int64_t beta2,
                 const torch::Tensor &dense_weights,
                 const torch::Tensor &row_offsets,
                 const torch::Tensor &col_val_ptr,
                 int64_t nnz,
                 const torch::Tensor &X,
                 int64_t _feature_flag,
                 const torch::Tensor &Y,
                 torch::Tensor &out) {
  uint32_t feature_flag = static_cast<uint32_t>(_feature_flag);
  int dev = dense_weights.get_device();

  // Choose which algorithm to use
  int row_offsets_len = row_offsets.sizes()[0];

  // TODO: Propagate error one layer up.
  int err = spqr_matvec(
      bits, m, n, beta1, beta2,
      nullptr,
      dense_weights.data_ptr(),
      row_offsets_len,
      row_offsets.data_ptr(),
      col_val_ptr.data_ptr(),
      nnz,
      X.data_ptr(),
      out.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev),
      nullptr,
      feature_flag);
}



// Function to convert an integer to half-precision using round-down
__half int2half_rd(const int value) {
  // Convert integer to float first
  float floatValue = static_cast<float>(value);
  // Convert float to __half
  __half halfValue = __float2half_rd(floatValue);
  return halfValue;
}

template<class Bit_t, class Scalar_t> Scalar_t host_dequantize(Bit_t q, Scalar_t s, Scalar_t z) {
  // TODO: Clean up these ifs.
  Scalar_t result;
  if constexpr (std::is_same_v<Scalar_t, half>) {
    result = s * (int2half_rd(static_cast<const int>(q)) - z);
  } else {
    result = s * (Scalar_t(q) - z);
  }
#if 0
  printf(" %f = %f x (%f - %f)\n", result, s, Scalar_t(q), z);
#endif
  return result;
}

template<class Weight_t> struct Weights2D {
  int m;
  int n;
  Weight_t *w;

  Weight_t &operator()(int i, int j) { return w[i * n + j]; }
};

#define UPDIV(X, Y) (((X) + (Y)-1) / (Y))

void torch_mul_timer(const torch::Tensor &deq_w,
                     const torch::Tensor &x,
                     torch::Tensor &y,
                     torch::Tensor &measurements) {
  int dev = deq_w.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  float *measurements_ptr = reinterpret_cast<float*>(measurements.data_ptr());

  Timer *timer = new Timer(stream);
  timer->start();

  // Make sure that the compiler doesn't optimize this away
  torch::mv_out(y, deq_w, x);

  cudaDeviceSynchronize();

  measurements_ptr[0] = timer->end();
  delete timer;
}

int torch_matvec(int m,
                 int n,
                 void *dequantized_w,
                 void *X,
                 void *y,
                 void *measurements,
                 cudaStream_t stream);

void dequantize_compressed(int m,
                           int n,
                           int bits,
                           int beta1,
                           int beta2,
                           const torch::Tensor &dense_weights,
                           // Outliers
                           // 32-bit
                           const torch::Tensor &row_offsets,
                           // 32-bit
                           const torch::Tensor &col_vals,
                           int nnz,
                           const torch::Tensor &deq_w_tensor) {
  half *deq_w = reinterpret_cast<half *>(deq_w_tensor.data_ptr());
  int tile_id{};
  int subtile_id{};
  int w_id{};
  uint64_t *dense_weights_ptr = reinterpret_cast<uint64_t *>(dense_weights.data_ptr());

  std::vector<float> deq_float32(m * n, 0);

  for (int ii = 0; ii < m; ii += beta1) {
    for (int jj = 0; jj < n; jj += beta2) {
      uint64_t w2_bits{};

      for (int k = 0; k < beta1; k++) {
        uint64_t partial = (dense_weights_ptr[k] >> (bits * (beta1 + 2)));
        w2_bits |= (partial << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (k / (SECOND_ORDER_FRAGMENT_SIZE_BITS / 4))));
      }

      SecondOrder w2{.v = w2_bits};

      const half wss2 = w2.members.ss.x;
      const half wsz2 = w2.members.zz.x;
      const half wzs2 = w2.members.ss.y;
      const half wzz2 = w2.members.zz.y;

      tile_id++;

      for (int i = 0; i < beta1 && i + ii < m; i++) {
        _BitArray wbits(dense_weights_ptr, bits);

        auto ws = int2half_rd(wbits[0]);
        auto wz = int2half_rd(wbits[1]);

        half s = host_dequantize(ws, wss2, wsz2);
        half z = host_dequantize(wz, wzs2, wzz2);

        for (int j = 0; j < beta2 && j + jj < n; j++) {
          half w = host_dequantize<half, half>(int2half_rd(wbits[2 + j]), s, z);
          deq_float32[(i + ii) * n + j + jj] = __half2float(w);
          w_id++;
        }
        subtile_id++;
        dense_weights_ptr++;
      }
    }
  }


  if (nnz) {
    int row_offsets_len = row_offsets.sizes()[0];
    int *_row_offsets = row_offsets.data_ptr<int>();
    ColVal *_col_vals = reinterpret_cast<ColVal *>(col_vals.data_ptr());

    if (row_offsets_len == m + 1) {
      for (int r = 0; r < m; r++) {
        for (int j = _row_offsets[r]; j < _row_offsets[r + 1]; j++) {
          auto c = _col_vals[j].members.c;
          deq_float32[r * n + c] += __half2float(_col_vals[j].members.v);
        }
      }
    } else {
      for (int r = 0; r < row_offsets_len - 1; r++) {
        for (int j = _row_offsets[r]; j < _row_offsets[r + 1]; j++) {
          auto c = _col_vals[j].members.c;
          int ptr = (r * 16 + j % 16) * n + c;
          deq_float32[ptr] += __half2float(_col_vals[j].members.v);
        }
      }
    }
  }

  for (int i = 0; i < m * n; i++) {
    deq_w[i] = __float2half_rn(deq_float32[i]);
  }
}

void spqr_mul_timer(int m,
                    int n,
                    // W and meta
                    int bits,
                    // Quantization
                    int beta1,
                    int beta2,
                    const torch::Tensor &weights,
                    // 16-bit
                    const torch::Tensor &row_offsets,
                    // 32-bit
                    const torch::Tensor &col_val,
                    int nnz,
                    // 16-bit
                    const torch::Tensor &X,
                    torch::Tensor &Y,
                    torch::Tensor &measurements,
                    uint32_t feature_flag) {
  int dev = weights.get_device();

  // Choose which algorithm to use
  int row_offsets_len = row_offsets.sizes()[0];

  int err = spqr_matvec(bits,
                        m,
                        n,
                        beta1,
                        beta2,
                        nullptr,
                        weights.data_ptr(),
                        row_offsets_len,
                        row_offsets.data_ptr(),
                        col_val.data_ptr(),
                        nnz,
                        X.data_ptr(),
                        Y.data_ptr(),
                        at::cuda::getCurrentCUDAStream(dev),
                        measurements.data_ptr(),
                        feature_flag);
}

enum class SparseCompressionStrategy {
  CSR = 0,
  PTCSR = 1
};

void tensor_compress_interleaved(
    int m,
    int n,
    int bits,
    const torch::Tensor &W,
    int beta1,
    int beta2,
    const torch::Tensor &W_s,
    const torch::Tensor &W_z,
    const torch::Tensor &W_s_s,
    const torch::Tensor &W_s_z,
    const torch::Tensor &W_z_s,
    const torch::Tensor &W_z_z,
    const torch::Tensor &row_offsets,
    const torch::Tensor &row_offsets_output,
    const torch::Tensor &col_vals,
    const torch::Tensor &col_vals_interleaved,
    const int sparse_strategy_compression,
    const torch::Tensor &out) {
  TORCH_CHECK(W.dtype() == torch::kChar, "W should be of type char")
  TORCH_CHECK(W_s.dtype() == torch::kChar, "W_s should be of type char")
  TORCH_CHECK(W_z.dtype() == torch::kChar, "W_z should be of type char")
  TORCH_CHECK(W_s_s.dtype() == torch::kHalf, "W_s_s should be of type half")
  TORCH_CHECK(W_s_z.dtype() == torch::kHalf, "W_s_z should be of type half")
  TORCH_CHECK(W_z_s.dtype() == torch::kHalf, "W_z_s should be of type half")
  TORCH_CHECK(W_z_z.dtype() == torch::kHalf, "W_z_z should be of type half")

  char *w = static_cast<char *>(W.data_ptr());
  int *r = static_cast<int *>(row_offsets.data_ptr());
  ColVal *cv = static_cast<ColVal *>(col_vals.data_ptr());


  if (sparse_strategy_compression == 1) {
    int *r_output = static_cast<int *>(row_offsets_output.data_ptr());
    ColVal *cv_interleaved = static_cast<ColVal *>(col_vals_interleaved.data_ptr());
    *r_output = 0;
    auto cv_interleaved_ptr = cv_interleaved;
    int count = 0;
    for (int i = 0; i < m; i += beta1) {
      auto block_interleaved_ptr = cv_interleaved_ptr;

      int _count = 0;
      for (int j = 0; j < beta1; j++) {
        _count = std::max(_count, r[i + j + 1] - r[i + j]);
      }

      auto row_ptr = block_interleaved_ptr;
      for (int j = 0; j < beta1; j++) {
        auto cv_ptr = row_ptr;
        for (int k = r[i + j]; k < r[i + j + 1]; k++) {
          cv_ptr->_ = cv[k]._;
          cv_ptr += beta1;
        }
        row_ptr++;
      }
      cv_interleaved_ptr += _count * beta1;
      count += _count * beta1;
      *(++r_output) = count;
    }
  }


  using Bit_t = uint64_t;

  constexpr int BITS = 3;
  TileArray<Bit_t, BITS> tile_array(static_cast<Bit_t *>(out.data_ptr()));

  int tile_m = UPDIV(m, beta1);

  half *w_s_s_ptr = reinterpret_cast<half *>(W_s_s.data_ptr());
  half *w_s_z_ptr = reinterpret_cast<half *>(W_s_z.data_ptr());
  half *w_z_s_ptr = reinterpret_cast<half *>(W_z_s.data_ptr());
  half *w_z_z_ptr = reinterpret_cast<half *>(W_z_z.data_ptr());

  char *w_s_ptr = reinterpret_cast<char *>(W_s.data_ptr());
  char *w_z_ptr = reinterpret_cast<char *>(W_z.data_ptr());


  for (int i = 0; i < m; i += beta1) {
    for (int j = 0; j < n; j += beta2) {
      int tile_i = i / beta1;
      int tile_j = j / beta2;

      int tile_id = tile_j * tile_m + tile_i;
      half2 ss;
      ss.x = w_s_s_ptr[tile_id];
      ss.y = w_z_s_ptr[tile_id];
      half2 zz;
      zz.x = w_s_z_ptr[tile_id];
      zz.y = w_z_z_ptr[tile_id];

      SecondOrder second_order{
        .members = {
          .ss = ss,
          .zz = zz
        }
      };
      uint64_t v = second_order.v;

      int to_add{};
      int k{};
      for (k = 0; k < beta1 && i + k < m; k++) {
        Bit_t tile[16] = {0};

        int id{};

        int s_id = tile_j * tile_m * beta1 + tile_i * beta1 + k;

        int ws = static_cast<int>(w_s_ptr[s_id]);
        int wz = static_cast<int>(w_z_ptr[s_id]);

        for (int l = 0; l < beta2 && j + l < n; l++, id++) {
          tile[id] = static_cast<Bit_t>(w[(i + k) * n + (j + l)]);
          if (!k) {
            to_add++;
          }
        }

        uint64_t PARTIAL_OFFSET = BITS * (beta2 + 2);

        uint64_t FRAG_MASK = Bit_t((1ull << SECOND_ORDER_FRAGMENT_SIZE_BITS) - 1ull);

        uint64_t partial =
            (v >> (Bit_t((k / (SECOND_ORDER_FRAGMENT_SIZE_BITS / 4))) * Bit_t(SECOND_ORDER_FRAGMENT_SIZE_BITS))) &
            FRAG_MASK;

        tile_array.push(ws, wz, tile, to_add, (partial << PARTIAL_OFFSET));
      }

      Bit_t tile[16] = {0};
      for (; k < beta1; k++) {
        tile_array.push(0, 0, tile, beta2);
      }
    }
  }
}

void spqr_mul_fused(int64_t m,
                 int64_t n,
                 int64_t bits,
                 int64_t beta1,
                 int64_t beta2,
                 const torch::Tensor &in_order,
                 const torch::Tensor &dense_weights,
                 const torch::Tensor &row_offsets,
                 const torch::Tensor &col_val_ptr,
                 int64_t nnz,
                 const torch::Tensor &X,
                 int64_t _feature_flag,
                 const torch::Tensor &Y,
                 torch::Tensor &out) {
  uint32_t feature_flag = static_cast<uint32_t>(_feature_flag);
  int dev = dense_weights.get_device();

  // Choose which algorithm to use
  int row_offsets_len = row_offsets.sizes()[0];

  // TODO: Propagate error one layer up.
  int err = spqr_matvec(
      bits, m, n, beta1, beta2, in_order.data_ptr(), dense_weights.data_ptr(),
      row_offsets_len, row_offsets.data_ptr(), col_val_ptr.data_ptr(), nnz,
      X.data_ptr(), out.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev), nullptr, feature_flag);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spqr_mul_timer", &spqr_mul_timer, "SPQR matvec.");
  m.def("dequantize_compressed", &dequantize_compressed, "SPQR dequantize compressed.");
  m.def("torch_mul_timer", &torch_mul_timer, "Torch matvec FP16 device.");
  m.def("tensor_compress_interleaved", &tensor_compress_interleaved, "Tensor compress.");
  m.def("spqr_mul", &spqr_mul, "SPQR matvec.");
  m.def("spqr_mul_fused", &spqr_mul_fused, "");
}

