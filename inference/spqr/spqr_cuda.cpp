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

#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/python.h>
#include <vector>

// Function to convert an integer to half-precision using round-down
__half int2half_rd(const int value) {
  // Convert integer to float first
  float floatValue = static_cast<float>(value);
  // Convert float to __half
  __half halfValue = __float2half_rd(floatValue);
  return halfValue;
}

template <class Bit_t, class Scalar_t>
Scalar_t host_dequantize(Bit_t q, Scalar_t s, Scalar_t z) {
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

template <class Weight_t> struct Weights2DConst {
  int m;
  int n;
  const Weight_t *w;

  Weight_t operator()(int i, int j) const { return w[i * n + j]; }
};

template <class Weight_t> struct Weights2D {
  int m;
  int n;
  Weight_t *w;
  Weight_t &operator()(int i, int j) { return w[i * n + j]; }
};

#define UPDIV(X, Y) (((X) + (Y)-1) / (Y))

int spqr_cuda(
    // W and meta
    const void *W, int prob_m, int prob_n,
    // Quantization
    int beta1, int beta2,
    // W 1st order stats
    void *W_s, void *W_z,
    // W 2nd order stats
    void *W_s_s, void *W_s_z, void *W_z_s, void *W_z_z, const void *X,
    // Outliers
    void *values,
    // 16-bit
    void *row_offsets,
    // 32-bit
    void *col_ptr,
    // 16-bit
    void *workspace, int groupsize = -1, int dev = 0, cudaStream_t stream = 0,
    int thread_m = -1, int thread_n = -1, int sms = -1, int max_par = 16);

namespace torch_test {
int torch_matvec(int m, int n, void *dequantized_w, void *X, void *y,
                 void *measurements, cudaStream_t stream);
}

int spqr_matvec(
    // W and meta
    int bits, int prob_m, int prob_n,
    // Quantization
    int beta1, int beta2, const void *buff0, const void *buff1, void *row_ids,
    void *row_offsets,
    // 32-bit
    void *col_vals, int nnz, int dense_row_count,
    // 16-bit
    // Input
    void *X,
    void *order,
    // Output
    void *y,
    // GPU meta
    cudaStream_t stream = nullptr, void *measurements = nullptr,
    uint32_t feature_flag = 0);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void torch_mul_device(int m, int n, const torch::Tensor &deq_w,
                      const torch::Tensor &X, torch::Tensor &Y,
                      torch::Tensor &measurements) {
  int dev = deq_w.get_device();
  torch_test::torch_matvec(m, n, deq_w.data_ptr(), X.data_ptr(), Y.data_ptr(),
                           measurements.data_ptr(),
                           at::cuda::getCurrentCUDAStream(dev));
}

void dequantize_compressed(int m, int n, int bits, int beta1, int beta2,
                           const torch::Tensor &buff0,
                           const torch::Tensor &buff1,
                           // Outliers
                           // 32-bit
                           const torch::Tensor &row_offsets,
                           // 32-bit
                           const torch::Tensor &col_vals, int nnz,
                           const torch::Tensor &deq_w_tensor) {

  SecondOrder *w2 = reinterpret_cast<SecondOrder *>(buff1.data_ptr());

  half *deq_w = reinterpret_cast<half*>(deq_w_tensor.data_ptr());
  int tile_id{};
  int subtile_id{};
  int w_id{};
  uint64_t *buff0_ptr = reinterpret_cast<uint64_t*>(buff0.data_ptr());

  for (int ii = 0; ii < m; ii += beta1) {
    for (int jj = 0; jj < n; jj += beta2) {
      const half wss2 = w2[tile_id].members.ss.x;
      const half wsz2 = w2[tile_id].members.zz.x;
      const half wzs2 = w2[tile_id].members.ss.y;
      const half wzz2 = w2[tile_id].members.zz.y;
      tile_id++;

      for (int i = 0; i < beta1 && i + ii < m; i++) {
        _BitArray wbits(buff0_ptr, bits);

        auto ws = int2half_rd(wbits[0]);
        auto wz = int2half_rd(wbits[1]);

        half s = host_dequantize(ws, wss2, wsz2);
        half z = host_dequantize(wz, wzs2, wzz2);

        for (int j = 0; j < beta2 && j + jj < n; j++) {
          half w = host_dequantize<half, half>(
              int2half_rd(wbits[2 + j]), s, z);
          deq_w[(i + ii) * n + j + jj] = w;
          w_id++;
        }
        subtile_id++;
        buff0_ptr++;
      }
    }
  }

  if (nnz) {
    int *_row_offsets = row_offsets.data_ptr<int>();
    ColVal *_col_vals = reinterpret_cast<ColVal *>(col_vals.data_ptr());
    for (int r = 0; r < m; r++) {
      for (int j = _row_offsets[r]; j < _row_offsets[r + 1]; j++) {
        auto c = _col_vals[j].members.c;
        deq_w[r * n + c] = deq_w[r * n + c] + _col_vals[j].members.v;
      }
    }
  }
}

inline int compress(int bits /* = 3 */, int bucket_value_count /* = 10 */,
                    const std::vector<uint8_t> &in, int *out) {
  int bit_buffer = 0;
  int id{};
  for (int i = 0; i < in.size(); i++) {
    if (i && i % bucket_value_count == 0) {
      out[id++] = bit_buffer;
      bit_buffer = 0;
    }
    bit_buffer |= in[i] << ((i % bucket_value_count) * bits);
  }
  if (in.size() % bucket_value_count != 0) {
    out[id++] = bit_buffer;
  }
  return id;
}

void tensor_compress2(const torch::Tensor &W, int m, int n, int bit_count,
                      int beta1, int beta2, torch::Tensor &out) {
  Weights2D<uint8_t> weights{
      .m = m, .n = n, .w = (uint8_t *)W.data_ptr<uint8_t>()};

  std::vector<uint8_t> arr(beta1 * beta2);

  constexpr int BUCKET_SIZE = 32;
  int w_bit_count = beta1 * beta2 * bit_count;
  int *buff = out.data_ptr<int>();
  for (int i = 0; i < m; i += beta1) {
    for (int j = 0; j < n; j += beta2) {
      for (int ii = 0, sub_wid = 0; ii < beta1; ii++) {
        for (int jj = 0; jj < beta2; jj++, sub_wid++) {
          auto w = weights(i + ii, j + jj);
          arr[sub_wid] = w;
        }
      }
      int offset = compress(bit_count, BUCKET_SIZE / bit_count, arr, buff);
      buff = buff + offset;
    }
  }
}

void tensor_compress(torch::Tensor W, int bit_count, torch::Tensor &out) {
  W = W.flatten();
  int value_count = W.size(0);

  // Values per bucket
  int values_per_bucket = 32 / bit_count;

  int total_buckets = out.size(0);

  // Position within bucket
  int buff_id = 0;

  // Current bucket
  int bucket_id = 0;

  // Current bucket
  int bucket = 0;

  for (int i = 0; i < value_count; ++i) {
    bucket |= W[i].item<int>() << (buff_id * bit_count);
    buff_id += 1;

    if (buff_id == values_per_bucket) {
      out[bucket_id] = bucket;
      buff_id = 0;
      bucket_id += 1;
      bucket = 0;
    }
  }

  if (buff_id > 0) {
    out[bucket_id] = bucket;
  }
}

void spqr_mul_timer(int m, int n,
                    // W and meta
                    int bits,
                    // Quantization
                    int beta1, int beta2,
                    // W 1st order stats
                    const torch::Tensor &raw_data,
                    // W 1st order stats
                    const torch::Tensor &second_order_data,
                    const torch::Tensor &row_ids,
                    // 16-bit
                    const torch::Tensor &row_offsets,
                    // 32-bit
                    const torch::Tensor &col_val, int nnz,
                    int dense_row_count,
                    // 16-bit
                    const torch::Tensor &X,
                    torch::Tensor &Y,
                    torch::Tensor &measurements, uint32_t feature_flag) {
  int dev = raw_data.get_device();
  int err = spqr_matvec(bits, m, n, beta1, beta2, raw_data.data_ptr(),
                        second_order_data.data_ptr(),
                        // CSR Outliers
                        row_ids.data_ptr(), row_offsets.data_ptr(),
                        col_val.data_ptr(), nnz, dense_row_count, X.data_ptr(),
                        nullptr, Y.data_ptr(), at::cuda::getCurrentCUDAStream(dev),
                        measurements.data_ptr(), feature_flag);
}

void spqr_mul(int m, int n,
              // W and meta
              int bits,
              // Quantization
              int beta1, int beta2,
              // W 1st order stats
              const torch::Tensor &buff0,
              // W 1st order stats
              const torch::Tensor &buff1,
              // 32-bit
              const torch::Tensor &row_ids, const torch::Tensor &row_offsets,
              // 16-bit
              const torch::Tensor &col_val_ptr,
              int nnz,
              int dense_row_count,
              // 16-bit
              const torch::Tensor &X, torch::Tensor &Y,
              uint32_t feature_flag = 0) {
  int dev = buff0.get_device();
  int err = spqr_matvec(
      bits, m, n, beta1, beta2, buff0.data_ptr(), buff1.data_ptr(),
      row_ids.data_ptr(), row_offsets.data_ptr(), col_val_ptr.data_ptr(), nnz,
      dense_row_count, X.data_ptr(), nullptr, Y.data_ptr(),
      at::cuda::getCurrentCUDAStream(dev), nullptr, feature_flag);
}

void tensor_compress_interleaved(
    int m, int n, int bits, const torch::Tensor &W, int beta1, int beta2,
    const torch::Tensor &W_s, const torch::Tensor &W_z,
    const torch::Tensor &W_s_s, const torch::Tensor &W_s_z,
    const torch::Tensor &W_z_s, const torch::Tensor &W_z_z,
    const torch::Tensor &out, const torch::Tensor &out_second_order) {
  TORCH_CHECK(W.dtype() == torch::kChar, "W should be of type char")
  TORCH_CHECK(W_s.dtype() == torch::kChar, "W_s should be of type char")
  TORCH_CHECK(W_z.dtype() == torch::kChar, "W_z should be of type char")
  TORCH_CHECK(W_s_s.dtype() == torch::kHalf, "W_s_s should be of type half")
  TORCH_CHECK(W_s_z.dtype() == torch::kHalf, "W_s_z should be of type half")
  TORCH_CHECK(W_z_s.dtype() == torch::kHalf, "W_z_s should be of type half")
  TORCH_CHECK(W_z_z.dtype() == torch::kHalf, "W_z_z should be of type half")

  char *w = static_cast<char *>(W.data_ptr());

  using Bit_t = uint64_t;

  constexpr int BITS = 3;
  TileArray<Bit_t, BITS> tile_array(static_cast<Bit_t *>(out.data_ptr()));

  SecondOrder *second_vector_ptr = static_cast<SecondOrder *>(out_second_order.data_ptr());

  int tile_m = UPDIV(m, beta1);
  int tile_n = UPDIV(n, beta2);

  int global_id{};

  for (int i = 0; i < m; i += beta1) {
    for (int j = 0; j < n; j += beta2) {
      int tile_i = i / beta1;
      int tile_j = j / beta2;

      int tile_id = tile_j * tile_m + tile_i;

      half *w_s_s = static_cast<half *>(W_s_s.data_ptr()) + tile_id;
      half *w_s_z = static_cast<half *>(W_s_z.data_ptr()) + tile_id;
      half *w_z_s = static_cast<half *>(W_z_s.data_ptr()) + tile_id;
      half *w_z_z = static_cast<half *>(W_z_z.data_ptr()) + tile_id;

      half2 ss; ss.x = *w_s_s; ss.y = *w_z_s;
      half2 zz; zz.x = *w_s_z; zz.y = *w_z_z;

      second_vector_ptr->members.ss = ss;
      second_vector_ptr->members.zz = zz;

      int to_add{};
      int k{};
      Bit_t sanity{};
      uint64_t v = second_vector_ptr->v;
      for (k = 0; k < beta1 && i + k < m; k++) {
        Bit_t tile[16]{};
        for (int ii = 0; ii < 16; ii++) tile[i] = 0;

        int id{};

        int s_id = tile_j * tile_m * beta1 + tile_i * beta1 + k;

        char *w_s = static_cast<char *>(W_s.data_ptr()) + s_id;
        char *w_z = static_cast<char *>(W_z.data_ptr()) + s_id;

        int ws = *w_s;
        int wz = *w_z;

        for (int l = 0; l < beta2 && j + l < n; l++, id++, global_id++) {
           tile[id] = static_cast<Bit_t>(w[(i + k) * n + (j + l)]);
          if (!k) {
            to_add++;
          }
        }
        uint64_t PARTIAL_OFFSET = BITS * (beta2 + 2); // = 54, for example

        uint64_t FRAG_MASK = Bit_t((1ull << SECOND_ORDER_FRAGMENT_SIZE_BITS) - 1ull);
        
        uint64_t partial = (v >> (Bit_t((k / (SECOND_ORDER_FRAGMENT_SIZE_BITS / 4))) * Bit_t(SECOND_ORDER_FRAGMENT_SIZE_BITS))) & FRAG_MASK;

        
        tile_array.push(ws, wz, tile, to_add, (partial << PARTIAL_OFFSET));

        sanity |= (partial << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (k / (SECOND_ORDER_FRAGMENT_SIZE_BITS / 4))));
      }

#if 1
      if (sanity != second_vector_ptr->v) {
        printf("valid = %d\n", sanity == second_vector_ptr->v);
        exit(1);
      }
#endif

      second_vector_ptr++;

      Bit_t tile[16]{};
      for (; k < beta1; k++) {
        tile_array.push(0, 0, tile, beta2);
      }
    }
  }
}

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed with error (%d) at line %d\n", status,           \
             __LINE__);                                                        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed with error (%d) at line %d\n", status,       \
             __LINE__);                                                        \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

int preprocess_sparse(int prob_m, int prob_n,
                      // Outliers
                      const torch::Tensor &values,
                      // 16-bit
                      const torch::Tensor &row_offsets,
                      // 32-bit
                      const torch::Tensor &col_ptr,
                      // 16-bit
                      const torch::Tensor &X, torch::Tensor &Y, int nnz,
                      torch::Tensor &cusparse_buffer) {
  half alpha = __float2half(1.f);
  half beta = __float2half(1.f);

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, prob_n, X.data_ptr(), CUDA_R_32F))
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, prob_m, Y.data_ptr(), CUDA_R_32F))

  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, prob_m, prob_n, nnz, row_offsets.data_ptr(), col_ptr.data_ptr(),
      values.data_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
  // Create sparse matrix A in CSR format
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
}

#if PYBIND_SKIP == 1
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spqr_mul", &spqr_mul, "SPQR gemv.");
  m.def("spqr_mul_timer", &spqr_mul_timer, "SPQR gemv.");
  m.def("spqr_dequantize_compressed", &dequantize_compressed,
        "SPQR dequantize compressed.");
  m.def("torch_mul_fp16", &torch_mul_device, "Torch matvec FP16 device.");
  m.def("tensor_compress", &tensor_compress, "Tensor compress.");
  m.def("tensor_compress2", &tensor_compress2, "Tensor compress.");
  m.def("tensor_compress_interleaved", &tensor_compress_interleaved,
        "Tensor compress.");
  m.def("preprocess_sparse", &preprocess_sparse, "Preprocess sparse.");
}
#endif

using namespace torch;
#if 1

int main() { return 0; }

#else

int main() {
  int m = 32;
  int n = 16;
  int beta1 = 16;
  int beta2 = 16;
  int bits = 3;

  using ValueType = float;

  auto c = [&](Tensor E) {
    int values_per_bucket = 32 / bits;
    int value_count = E.size(0);
    int total_buckets =
        (value_count + values_per_bucket - 1) / values_per_bucket;
    auto out = torch::zeros(total_buckets, torch::kInt32);
    tensor_compress2(E, m, n, bits, beta1, beta2, out);
    return out;
  };
  auto updiv = [](const auto x, const auto y) { return (x + y - 1) / y; };

  auto num_first_order_groups = updiv(m, beta1) * n;
  torch::Tensor deq_W = torch::zeros({m, n}, torch::kFloat);

  // torch::Tensor W = c(torch::reshape(torch::arange({m * n}, torch::kInt), {m,
  // n}));
  torch::Tensor W = torch::ones({m * n}, torch::kInt);

  torch::Tensor W_s = c(torch::ones({num_first_order_groups}, torch::kInt));
  torch::Tensor W_z = c(torch::zeros({num_first_order_groups}, torch::kInt));

  int num_second_order_groups = updiv(m, beta1) * updiv(n, beta2);
  torch::Tensor W_s_s = torch::ones({num_second_order_groups}, torch::kFloat);
  torch::Tensor W_s_z = torch::zeros({num_second_order_groups}, torch::kFloat);
  torch::Tensor W_z_s = torch::ones({num_second_order_groups}, torch::kFloat);
  torch::Tensor W_z_z = torch::zeros({num_second_order_groups}, torch::kFloat);

  torch::Tensor values = torch::zeros({0}, torch::kFloat);
  torch::Tensor row_offsets = torch::zeros({m + 1}, torch::kInt);
  torch::Tensor col_ptr = torch::zeros({0}, torch::kShort);
  torch::Tensor x = torch::arange({n}, torch::kFloat);
  torch::Tensor y = torch::zeros({m}, torch::kFloat);
  torch::Tensor y_gt = torch::zeros({m}, torch::kFloat);
  int nnz = 0;

  spqr_mul_host<ValueType, uint64_t>(
      m, n, bits, W, beta1, beta2, W_s, W_z, W_s_s, W_s_z, W_z_s, W_z_z, values,
      row_offsets, col_ptr, nnz, x, y_gt, y, deq_W);

  std::cout << "deq_w=\n"
            << deq_W << "\ny=\n"
            << y << "\ny_gt=" << y_gt << std::endl;

  return 0;
}

#endif