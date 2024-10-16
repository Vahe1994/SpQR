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
#include <ATen/cuda/Exceptions.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#define DEVICE_INLINE __forceinline__ __device__


extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);


template<class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

DEVICE_INLINE uint64_t recover_second_order(uint64_t val) {
  constexpr unsigned int FULL_MASK = 0xffffffffu;
  val |= __shfl_xor_sync(FULL_MASK, val, 2);
  val |= __shfl_xor_sync(FULL_MASK, val, 4);
  val |= __shfl_xor_sync(FULL_MASK, val, 8);
  return val;
}

using u64 = unsigned long long;
using s32 = int;
using u32 = unsigned int;
using u16 = unsigned short;

union RowBits {
  uint64_t mask;

  struct {
    uint64_t s: 3;
    uint64_t z: 3;
    uint64_t w: 48;
  };

  __device__ __forceinline__ u16 get_w(int i) const {
    return (w >> (i * 3u)) & ((1u << 3u) - 1u);
  }

  __device__ __forceinline__ u32 get_w2(int i) const {
    return (mask >> (i * 6u)) & ((1u << 6u) - 1u);
  }
};

half2 DEVICE_INLINE dequantize2(const half2 &q,
                                const half2 &s,
                                const half2 &z) {
  const half2 &res = __hmul2(s, __hsub2(q, z));
  return res;
}

template<class Bit_t, class Scalar_t> DEVICE_INLINE Scalar_t dequantize(Bit_t q,
                                                                        Scalar_t s,
                                                                        Scalar_t z) {
  if constexpr (std::is_same<Bit_t, half>::value) {
    return __hmul(s, __hsub(q, z));
  } else {
    return __hmul(s, __hsub(__uint2half_rd(q, z)));
  }
}

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y)-1) / (Y))
#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))

[[nodiscard]] __device__ __host__ CUINLINE int updiv(int x, int y) {
  return (x + y - 1) / y;
}

struct Timer {
  cudaEvent_t ce_start{}, ce_stop{};
  cudaStream_t stream;

  void start() { AT_CUDA_CHECK(cudaEventRecord(ce_start, stream)); }

  float end() {
    float time;
    AT_CUDA_CHECK(cudaEventRecord(ce_stop, 0));
    AT_CUDA_CHECK(cudaEventSynchronize(ce_stop));
    AT_CUDA_CHECK(cudaEventElapsedTime(&time, ce_start, ce_stop));
    // Returns ms
    return time;
  }

  Timer(cudaStream_t stream) : stream(stream) {
    AT_CUDA_CHECK(cudaEventCreate(&ce_start));
    AT_CUDA_CHECK(cudaEventCreate(&ce_stop));
  }

  Timer(Timer &&timer) = delete;

  Timer(const Timer &timer) = delete;

  ~Timer() {
    AT_CUDA_CHECK(cudaEventDestroy(ce_start));
    AT_CUDA_CHECK(cudaEventDestroy(ce_stop));
  }
};


template<typename T> __device__ T _debug_halfs(T v) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f\n", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f\n", __half2float(v.x), __half2float(v.y));
  }
  return v;
}

template<typename T, typename... Arguments> __device__ void _debug_halfs(T v, Arguments... vals) {
  if constexpr (std::is_same<T, half>::value) {
    printf(" %f", __half2float(v));
  } else if constexpr (std::is_same<T, half2>::value) {
    printf(" %f %f", __half2float(v.x), __half2float(v.y));
  }
  _debug_halfs(vals...);
}

template<class Scalar_t> __host__ __device__ auto vectorize(Scalar_t *ptr) {
  if constexpr (std::is_same<Scalar_t, float>::value) {
    return reinterpret_cast<float2 *>(ptr);
  } else if constexpr (std::is_same<Scalar_t, half>::value) {
    return reinterpret_cast<half2 *>(ptr);
  } else {
    return ptr;
  }
}

template<class Vec_t> __host__ __device__ auto scalarize(void *ptr) {
  if constexpr (std::is_same<Vec_t, float>::value ||
                std::is_same<Vec_t, float2>::value) {
    return reinterpret_cast<float *>(ptr);
  } else if constexpr (std::is_same<Vec_t, half2>::value) {
    return reinterpret_cast<half *>(ptr);
  } else {
    return ptr;
  }
}

DEVICE_INLINE float add_and_accum(float a, float b) {
  return a + b;
}

DEVICE_INLINE half add_and_accum(const half2 &a, const half2 &b) {
  half2 r = __hadd2(a, b);
  return __hadd(r.x, r.y);
}

template<class T> DEVICE_INLINE u16 get_col(T m) {
  return static_cast<u16>(m & T((1u << 16u) - 1u));
}

DEVICE_INLINE half get_val(u32 m) {
  u16 _v = m >> 16u;
  half v = *reinterpret_cast<half *>(&_v);
  return v;
}


#define CALL_DENSE(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH) \
    constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT; \
    constexpr int BLOCK_WIDTH = _BLOCK_WIDTH; \
    size_t smem_size = sizeof(half2) * (BLOCK_WIDTH * SHARED_OFFSET);                   \
    F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, float, uint64_t, PIPELINE_DEPTH> \
            <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), \
            dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, 1, 1), smem_size, \
            stream>>>(prob_m, \
            prob_n, \
            raw_data,                               \
            X_ptr, \
            order_ptr, \
            y_ptr);


#define CALL_FUSED(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH) \
    constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT; \
    constexpr int BLOCK_WIDTH = _BLOCK_WIDTH; \
    size_t smem_size = sizeof(half2) * prob_n / 2;                   \
    F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, uint64_t, PIPELINE_DEPTH> \
            <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), \
            dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, __min(updiv(prob_m, 16), BLOCK_HEIGHT), 1), smem_size, \
            stream>>>(prob_m, \
            prob_n, \
            raw_data_ptr,                               \
            X_ptr, \
            row_offsets_ptr, \
            col_vals_ptr, \
            order_ptr, \
            y_ptr);


static constexpr u32 SHARED_OFFSET = 32;

// Wait until at most `n` async copy stages are still pending.
template<int n> DEVICE_INLINE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n"::"n"(n));
}

template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class Acc_t, class W_t /* = uint64_t */, int
PIPELINE_DEPTH> __global__ void spqr_quantized_matvec_dense(
    // W and meta
    unsigned int prob_m,
    unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const half *__restrict__ x,
    // Outliers
    const short *__restrict__ order,
    // Output
    half *__restrict__ y_fp16) {
  /*
           ┌─────────────┐ ┌─┐   ┌─┐
   beta1   │   block 0   │ │ │   │ │
           ├─────────────┤ │ │   │ │
   beta1   │   block 1   │ │ │   │ │
           └─────────────┘ │x│ = │Y│
           │    ...      │ │ │   │ │
           ┌─────────────┐ │ │   │ │
   beta1   │  block m-1  │ │ │   │ │
           └─────────────┘ └─┘   └─┘
  */
  static constexpr int WARP_SIZE = 32;
  static constexpr u32 HALF_WARP_SIZE = WARP_SIZE / 2;
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  static constexpr u32 NUM_HALF_WARPS = BLOCK_HEIGHT * BLOCK_WIDTH;
  static constexpr u32 THREAD_COUNT = BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;

  extern __shared__ half2 s_x2[];
  __shared__ half2 s_half2_lut_global[64 * NUM_HALF_WARPS];
  __shared__ float s_y[OUTPUT_SIZE];
  __shared__ u32 s_row_offsets[OUTPUT_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  for (int i = thread_xy; i < 64 * NUM_HALF_WARPS; i += THREAD_COUNT) {
    s_half2_lut_global[i] = make_half2(
        __int2half_rd(i & 0b111),
        __int2half_rd((i >> 3) & 0b111)
    );
  }

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / HALF_WARP_SIZE) << 6);
  const half2 *x2 = reinterpret_cast<const half2 *>(x);


  if constexpr (std::is_same<Acc_t, float>::value) {
    if (threadIdx.x < BETA1) {
      // TOD: Check if this really sets s_y to zero.
      asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
          : "r"(__nvvm_get_smem_pointer(s_y + threadIdx.x))
          );
    }
  } else {
    if (threadIdx.x < BETA1 / 2) {
      asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
          : "r"(__nvvm_get_smem_pointer(s_y + threadIdx.x))
          );
    }
  }

  asm volatile ("cp.async.commit_group;");

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_spqr_tiles_per_cuda_block = UPDIV(prob_n, BETA2);

  u32 row_pos = threadIdx.x & 0xF; // threadIdx.x % BETA1;
  const u32 subtile_id = threadIdx.x / BETA1;

  const W_t *local_raw_data =
      raw_data + blockIdx.x * num_spqr_tiles_per_cuda_block * BETA1 + subtile_id * BETA1 + row_pos;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS = 18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  const auto s_x2_ = s_x2 + subtile_id * SHARED_OFFSET;


  cp_async_wait<0>();
  Acc_t acc{};
  __syncthreads();
  for (u32 i = subtile_id; i < num_spqr_tiles_per_cuda_block; i += NUM_SPQR_TILES_PER_ITERATION, local_raw_data +=
                                                                                                     NUM_SPQR_TILES_PER_ITERATION *
                                                                                                     BETA1) {
#if 0
    asm volatile ("cp.async.ca.shared.global [%0], [%1], 4 ;\n"::"r"(__nvvm_get_smem_pointer(s_x2 + subtile_id * SHARED_OFFSET + (threadIdx.x & 0xF) / 2)), "l"(x2 + i * BETA2 / 2 + (threadIdx.x & 0xF) / 2));
    asm volatile ("cp.async.commit_group;");
#else
    s_x2[subtile_id * SHARED_OFFSET + (threadIdx.x & 0xF) / 2] = x2[i * BETA2 / 2 + (threadIdx.x & 0xF) / 2];
#endif

    auto v = __ldg(local_raw_data);
    RowBits row_bits{
        .mask = v
    };
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order(s_order_partial)};


    half2 first_order_quantized = s_half2_lut[row_bits.get_w2(0)];
    half2 first_order_dequantized = dequantize2(first_order_quantized,
                                                _s.get_sws2(),
                                                _s.get_swz2());

    half2 ws2 = __half2half2(first_order_dequantized.x);
    half2 wz2 = __half2half2(first_order_dequantized.y);

#if 0
    cp_async_wait<0>();
#else
    __threadfence_block();
#endif

#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      if constexpr (std::is_same<Acc_t, float>::value) {
        half2 q = s_half2_lut[row_bits.get_w2(j + 1)];
        half2 w = dequantize2(q, ws2, wz2);
        float2 x_fp32 = __half22float2(s_x2_[j]);
        float2 w_fp32 = __half22float2(w);
        acc = fmaf(x_fp32.x, w_fp32.x, acc);
        acc = fmaf(x_fp32.y, w_fp32.y, acc);
      } else {
        int q_x = row_bits.get_w(2 * j);
        int q_y = row_bits.get_w(2 * j + 1);
        half2 q = make_half2(__int2half_rd(q_x), __int2half_rd(q_y));
        half2 w = dequantize2(q, ws2, wz2);
        acc = __hfma2(s_x2[i * BETA2 / 2 + j], w, acc);
      }
    }
  }

  if (!(subtile_id & 1 == 0 && threadIdx.x + BETA1 >= blockDim.x)) {
    auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
    acc = add_and_accum(other, acc);
  }

  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // TODO: Double check this
  auto *s_fp32_buff = reinterpret_cast<float *>(s_half2_lut_global + threadIdx.y * MAX(WARP_SIZE - 1, 1) * BETA1);

  u32 subwarp_id = threadIdx.x / WARP_SIZE;
  if (subwarp_id >= 1 && threadIdx.x % WARP_SIZE < BETA1) {
    s_fp32_buff[(subwarp_id - 1) * BETA1 + threadIdx.x % WARP_SIZE] = acc;
  }

  __syncthreads();

  if (!subtile_id && threadIdx.x < BETA1) {
    for (int i = 0; i < WARP_COUNT - 1; i++) {
      acc += s_fp32_buff[i * BETA1 + threadIdx.x];
    }
  }

  if (order == nullptr) {
    if (threadIdx.x < BETA1) {
      y_fp16[tile_row_id * BETA1 + threadIdx.x] = __float2half(acc);
    }
  } else {
    // TODO:
    if (threadIdx.x < BETA1) {
      short row = order[tile_row_id * BETA1 + threadIdx.x];
      y_fp16[row] = __float2half_rn(acc);
    }
  }
}

__device__ __forceinline__ uint32_t __ld_stream(const uint32_t *ptr) {
  uint32_t v;
  asm volatile(
      "{\n"
      "   ld.global.ca.u32 %0, [%1];\n"
      "}\n" : "=r"(v) : "l"(ptr)
      );
  return v;
}

template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class W_t /* = uint64_t */, int
PIPELINE_DEPTH> __global__ void spqr_quantized_matvec_fused_csr_modified(
    // W and meta
    unsigned int prob_m,
    unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const half *__restrict__ x,
    // Outliers
    const int *__restrict__ row_offsets_interleaved,
    const u32 *__restrict__ col_vals_interleaved,
    const short *__restrict__ order,
    // Output
    half *__restrict__ y_fp16) {
  /*
           ┌─────────────┐ ┌─┐   ┌─┐
   beta1   │   block 0   │ │ │   │ │
           ├─────────────┤ │ │   │ │
   beta1   │   block 1   │ │ │   │ │
           └─────────────┘ │x│ = │Y│
           │    ...      │ │ │   │ │
           ┌─────────────┐ │ │   │ │
   beta1   │  block m-1  │ │ │   │ │
           └─────────────┘ └─┘   └─┘
  */
  static constexpr u32 WARP_SIZE = 32;
  static constexpr u32 HALF_WARP_SIZE = WARP_SIZE / 2;

  static constexpr u32 NUM_HALF_WARPS = BLOCK_HEIGHT * BLOCK_WIDTH;
  static constexpr u32 THREAD_COUNT = BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;

  extern __shared__ half2 s_x2[];
  __shared__ half2 s_half2_lut_global[64 * NUM_HALF_WARPS];
  __shared__ float s_y[OUTPUT_SIZE];
  __shared__ u32 s_row_offsets[2];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  for (int i = thread_xy; i < 64 * NUM_HALF_WARPS; i += THREAD_COUNT) {
    s_half2_lut_global[i] = make_half2(
        __int2half_rd(i & 0b111),
        __int2half_rd((i >> 3) & 0b111)
    );
  }

#if 0
  for (int i = thread_xy; i < 64 * NUM_HALF_WARPS; i += THREAD_COUNT) {
    reinterpret_cast<u32*>(s_half2_lut_global)[i] = (LUT[i & 0b111] | (LUT[(i >> 3) & 0b111]) << 16u);
subtile_id & (1 == 0) && threadIdx.x + BETA1 >= blockDim.x)) {
    auto other = __shfl_down_sync
    auto _gt = make_half2(
        __int2half_rd(i & 0b111),
        __int2half_rd((i >> 3) & 0b111)
    );

    auto gt = *reinterpret_cast<u32*>(&_gt);
    auto tst = (LUT[i & 0b111] | (LUT[(i >> 3) & 0b111]) << 16u);

    // printf("%u %u\n", gt, tst);
  }
#endif

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / HALF_WARP_SIZE) << 6);
  const half2 *x2 = reinterpret_cast<const half2 *>(x);


  const auto x2_count = prob_n >> 1;

  if (thread_xy < OUTPUT_SIZE) {
    // TOD: Check if this really sets s_y to zero.
    asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
        : "r"(__nvvm_get_smem_pointer(s_y + thread_xy))
        );
  }

  // TODO: Is this correct? Should x/y be swapped? Should threadidx be used here? - Should be correct, now
  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i < 2; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i, row_offsets_interleaved + blockIdx.x + i, sizeof(u32));
  }

  u32 pipeline_id, pipeline_stack_ptr{};
  for (pipeline_id = 0; pipeline_id < PIPELINE_DEPTH && (pipeline_id * THREAD_COUNT + thread_xy) < x2_count; pipeline_id
      ++) {
    u32 idx = pipeline_id * THREAD_COUNT + thread_xy;
    __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
    pipeline_stack_ptr++;
    __pipeline_commit();
  }

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_tiles_per_tile_row = UPDIV(prob_n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  u32 row_pos = thread_xy & 0xF;
  const u32 subtile_id = threadIdx.x / BETA1;

  auto raw_data_offset = tile_row_id * prob_n + threadIdx.x;
  const W_t *local_raw_data = raw_data + raw_data_offset;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS = 18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  float acc{};

  for (u32 i = subtile_id; i < num_tiles_per_tile_row;
       i += NUM_SPQR_TILES_PER_ITERATION, local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1) {
    auto v = __ldg(local_raw_data);
    RowBits row_bits{
        .mask = v
    };
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order(s_order_partial)};
    half2 first_order_quantized = s_half2_lut[row_bits.get_w2(0)];
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    half2 ws2 = __half2half2(first_order_dequantized.x);
    half2 wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);

    if (pipeline_stack_ptr > 0) {
      __pipeline_wait_prior(pipeline_stack_ptr - 1);
      pipeline_stack_ptr--;
    }

    __syncthreads();
#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = s_half2_lut[row_bits.get_w2(j + 1)];
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }

    unsigned idx = pipeline_id * THREAD_COUNT + thread_xy;
    if (idx < x2_count) {
      __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
      pipeline_id++;
      pipeline_stack_ptr++;
      __pipeline_commit();
    }
  }

  u32 s = s_row_offsets[0];
  u32 e = s_row_offsets[1];
  half *s_x = reinterpret_cast<half *>(s_x2);


  if (s + thread_xy < e) {
    ColVal colval{ ._ = col_vals_interleaved[s + thread_xy] };
    auto c = colval.members.c;
    auto v = colval.members.v;
    acc += __half2float(v) * __half2float(s_x[c]);
  }

  for (u32 i = s + thread_xy + BLOCK_WIDTH * BETA1; i < e; i += BLOCK_WIDTH * BETA1) {
    ColVal colval{
        ._ = col_vals_interleaved[i]
    };

    if (!colval._) break;

    auto c = colval.members.c;
    auto v = colval.members.v;
    acc += __half2float(v) * __half2float(s_x[c]);
  }

  if (!(subtile_id & 1 == 0 && threadIdx.x + BETA1 >= blockDim.x)) {
    auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
    acc = add_and_accum(other, acc);
  }

  // TODO: Double check this
  auto *s_fp32_buff = reinterpret_cast<float *>(s_half2_lut_global + threadIdx.y * MAX(WARP_SIZE - 1, 1) * BETA1);

  u32 subwarp_id = threadIdx.x / WARP_SIZE;
  if (subwarp_id >= 1 && threadIdx.x % WARP_SIZE < BETA1) {
    s_fp32_buff[(subwarp_id - 1) * BETA1 + threadIdx.x % WARP_SIZE] = acc;
  }

  __syncthreads();

  if (!subtile_id && threadIdx.x < BETA1) {
    for (int i = 0; i < WARP_COUNT - 1; i++) {
      acc += s_fp32_buff[i * BETA1 + threadIdx.x];
    }
  }

  if (order == nullptr) {
    if (threadIdx.x < BETA1) {
      y_fp16[tile_row_id * BETA1 + threadIdx.x] = __float2half(acc);
    }
  } else {
    // TODO:
    if (threadIdx.x < BETA1) {
      short row = order[tile_row_id * BETA1 + threadIdx.x];
      y_fp16[row] = __float2half_rn(acc);
    }
  }
}

template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class W_t /* = uint64_t */, int
PIPELINE_DEPTH> __global__ void spqr_quantized_matvec_fused_csr(
    // W and meta
    unsigned int prob_m,
    unsigned int prob_n,
    // W 1st order stats
    const W_t *__restrict__ raw_data,
    const half *__restrict__ x,
    // Outliers
    const int *__restrict__ row_offsets,
    const u32 *__restrict__ col_vals,
    const short *__restrict__ order,
    // Output
    half *__restrict__ y_fp16) {
  /*
           ┌─────────────┐ ┌─┐   ┌─┐
   beta1   │   block 0   │ │ │   │ │
           ├─────────────┤ │ │   │ │
   beta1   │   block 1   │ │ │   │ │
           └─────────────┘ │x│ = │Y│
           │    ...      │ │ │   │ │
           ┌─────────────┐ │ │   │ │
   beta1   │  block m-1  │ │ │   │ │
           └─────────────┘ └─┘   └─┘
  */
  static constexpr u32 WARP_SIZE = 32;
  static constexpr u32 HALF_WARP_SIZE = WARP_SIZE / 2;

  static constexpr u32 NUM_HALF_WARPS = BLOCK_HEIGHT * BLOCK_WIDTH;
  static constexpr u32 THREAD_COUNT = BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;

  extern __shared__ half2 s_x2[];
  __shared__ half2 s_half2_lut_global[64 * NUM_HALF_WARPS];
  __shared__ float s_y[OUTPUT_SIZE];
  __shared__ u32 s_row_offsets[OUTPUT_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  for (int i = thread_xy; i < 64 * NUM_HALF_WARPS; i += THREAD_COUNT) {
    s_half2_lut_global[i] = make_half2(
        __int2half_rd(i & 0b111),
        __int2half_rd((i >> 3) & 0b111)
    );
  }

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / HALF_WARP_SIZE) << 6);
  const half2 *x2 = reinterpret_cast<const half2 *>(x);


  const auto x2_count = prob_n >> 1;

  if (thread_xy < OUTPUT_SIZE) {
    // TOD: Check if this really sets s_y to zero.
    asm volatile ("cp.async.ca.shared.global [%0], [%0], 4, 0 ;\n" :
        : "r"(__nvvm_get_smem_pointer(s_y + thread_xy))
        );
  }

  // TODO: Is this correct? Should x/y be swapped? Should threadidx be used here? - Should be correct, now
  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= OUTPUT_SIZE; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i, row_offsets + blockIdx.x * BLOCK_HEIGHT * BETA1 + i, sizeof(u32));
  }

  u32 pipeline_id, pipeline_stack_ptr{};
  for (pipeline_id = 0; pipeline_id < PIPELINE_DEPTH && (pipeline_id * THREAD_COUNT + thread_xy) < x2_count; pipeline_id
      ++) {
    u32 idx = pipeline_id * THREAD_COUNT + thread_xy;
    __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
    pipeline_stack_ptr++;
    __pipeline_commit();
  }

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_tiles_per_tile_row = UPDIV(prob_n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  u32 row_pos = thread_xy & 0xF;
  const u32 subtile_id = threadIdx.x / BETA1;

  auto raw_data_offset = tile_row_id * prob_n + threadIdx.x;
  const W_t *local_raw_data = raw_data + raw_data_offset;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS = 18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  float acc{};

  for (u32 i = subtile_id; i < num_tiles_per_tile_row;
       i += NUM_SPQR_TILES_PER_ITERATION, local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1) {
    auto v = __ldg(local_raw_data);
    RowBits row_bits{
        .mask = v
    };
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order(s_order_partial)};
    half2 first_order_quantized = s_half2_lut[row_bits.get_w2(0)];
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    half2 ws2 = __half2half2(first_order_dequantized.x);
    half2 wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);

    if (pipeline_stack_ptr > 0) {
      __pipeline_wait_prior(pipeline_stack_ptr - 1);
      pipeline_stack_ptr--;
    }

    __syncthreads();
#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = s_half2_lut[row_bits.get_w2(j + 1)];
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }

    unsigned idx = pipeline_id * THREAD_COUNT + thread_xy;
    if (idx < x2_count) {
      __pipeline_memcpy_async(s_x2 + idx, x2 + idx, sizeof(half2));
      pipeline_id++;
      pipeline_stack_ptr++;
      __pipeline_commit();
    }
  }
  __syncthreads();

  auto s_y_scalar = scalarize<float>(s_y + threadIdx.y * BETA1);
  auto s_y_vectorized = vectorize(s_y_scalar);

#if 0
  u32 s = s_row_offsets[0];
  u32 e = s_row_offsets[BETA1];

  half *s_x = reinterpret_cast<half *>(s_x2);

  for (u32 i = s + threadIdx.x; i < e; i += BLOCK_WIDTH * BETA1) {
#else
  u32 t = threadIdx.y * BETA1 + row_pos;
  u32 s = s_row_offsets[t];
  u32 e = s_row_offsets[t + 1];

  half *s_x = reinterpret_cast<half *>(s_x2);

  for (u32 i = s + subtile_id; i < e; i += BLOCK_WIDTH) {
#endif

    ColVal colval{
        ._ = __ldg(col_vals + i)
    };
    auto c = colval.members.c;
    auto v = colval.members.v;
    acc += __half2float(v) * __half2float(s_x[c]);
  }

  __syncthreads();

  if (!(subtile_id & 1 == 0 && threadIdx.x + BETA1 >= blockDim.x)) {
    auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
    acc = add_and_accum(other, acc);
  }

  // TODO: Double check this
  auto *s_fp32_buff = reinterpret_cast<float *>(s_half2_lut_global + threadIdx.y * MAX(WARP_SIZE - 1, 1) * BETA1);

  u32 subwarp_id = threadIdx.x / WARP_SIZE;
  if (subwarp_id >= 1 && threadIdx.x % WARP_SIZE < BETA1) {
    s_fp32_buff[(subwarp_id - 1) * BETA1 + threadIdx.x % WARP_SIZE] = acc;
  }

  __syncthreads();

  if (!subtile_id && threadIdx.x < BETA1) {
    for (int i = 0; i < WARP_COUNT - 1; i++) {
      acc += s_fp32_buff[i * BETA1 + threadIdx.x];
    }
  }

  if (order == nullptr) {
    if (threadIdx.x < BETA1) {
      y_fp16[tile_row_id * BETA1 + threadIdx.x] = __float2half(acc);
    }
  } else {
    // TODO:
    if (threadIdx.x < BETA1) {
      short row = order[tile_row_id * BETA1 + threadIdx.x];
      y_fp16[row] = __float2half_rn(acc);
    }
  }
}

template<class T> const T &__min(const T &a, const T &b) {
  return (b < a) ? b : a;
}

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,     \
             cudaGetErrorString(status), status);                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

union Features {
  uint32_t _;

  struct {
    uint32_t is_fp32: 1;
    uint32_t dense_only: 1;
    uint32_t naive_sparse: 1;
    uint32_t torch: 1;
    uint32_t is_async: 1;
    uint32_t shared_sparse: 1;
    uint32_t single_sparse: 1;
    uint32_t cusparse: 1;
    uint32_t fused_sparse: 1;
    uint32_t shared_sparse_baseline: 1;
    uint32_t shared_mixture: 1;
    uint32_t rest: 21;
  } flags;
};

int spqr_matvec(
    // W and meta
    int bits,
    int prob_m,
    int prob_n,
    // Quantization
    int beta1,
    int beta2,
    const void *raw_data,
    // 32-bit
    int row_offsets_len,
    void *row_offsets,
    // 16-bit
    void *col_vals,
    int nnz,
    // 16-bit
    // Input
    void *X,
    void *order,
    // Output
    void *y,
    cudaStream_t stream,
    void *measurements,
    uint32_t feature_flag) {
  Timer *timer{};
  if (measurements) {
    timer = new Timer(stream);
    timer->start();
  }

  if (prob_m == 0 || prob_n == 0) {
    return 0;
  }

  Features features{._ = feature_flag};

  const uint64_t *raw_data_ptr = (const uint64_t *) raw_data;
  const half *X_ptr = (const half *) X;
  const int *row_offsets_ptr = (const int *) row_offsets;
  half *y_ptr = (half *) y;
  const auto *col_vals_ptr = (const u32 *) col_vals;
  const short *order_ptr = (const short *) order;

  int ret = 0;

  if (prob_m + 1 != row_offsets_len) {
    if (prob_m % 16 == 0 && prob_n % 256 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr_modified, 1, 16, 2);
    } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr_modified, 1, 8, 2);
    } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr_modified, 1, 4, 1);
    } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr_modified, 1, 2, 1);
    } else {
      CALL_FUSED(spqr_quantized_matvec_fused_csr_modified, 1, 1, 1);
    }
  } else {
    if (prob_m % 16 == 0 && prob_n % 256 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr, 1, 16, 2);
    } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr, 1, 8, 2);
    } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr, 1, 4, 1);
    } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
      CALL_FUSED(spqr_quantized_matvec_fused_csr, 1, 2, 1);
    } else {
      CALL_FUSED(spqr_quantized_matvec_fused_csr, 1, 1, 1);
    }

  }

  if (!features.flags.is_async) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  if (measurements) {
    static_cast<float *>(measurements)[0] = timer->end();
    delete timer;
  }

  return ret;
}
