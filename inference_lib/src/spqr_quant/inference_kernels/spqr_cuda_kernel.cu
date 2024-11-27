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

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define DEVICE_INLINE __forceinline__ __device__

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

template <class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

DEVICE_INLINE uint64_t recover_second_order_sync(uint64_t val) {
  unsigned int FULL_MASK = 0xffffffffu;
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
    uint64_t s : 3;
    uint64_t z : 3;
    uint64_t w : 48;
  };

  __device__ __forceinline__ u16 get_w(u32 i) const {
    return (w >> (i * 3u)) & ((1u << 3u) - 1u);
  }

  __device__ __forceinline__ u32 get_w2(u32 i) const {
    return (mask >> (i * 6u)) & ((1u << 6u) - 1u);
  }
};

half2 DEVICE_INLINE dequantize2(const half2 &q, const half2 &s,
                                const half2 &z) {
  const half2 &res = __hmul2(s, __hsub2(q, z));
  return res;
}

template <class Bit_t, class Scalar_t>
DEVICE_INLINE Scalar_t dequantize(Bit_t q, Scalar_t s, Scalar_t z) {
  if constexpr (std::is_same<Bit_t, half>::value) {
    return __hmul(s, __hsub(q, z));
  } else {
    return __hmul(s, __hsub(__uint2half_rd(q, z)));
  }
}

#define CUINLINE __forceinline__

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))
#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))

[[nodiscard]] __device__ __host__ CUINLINE int updiv(int x, int y) {
  return (x + y - 1) / y;
}

template <class Scalar_t> __host__ __device__ auto vectorize(Scalar_t *ptr) {
  if constexpr (std::is_same<Scalar_t, float>::value) {
    return reinterpret_cast<float2 *>(ptr);
  } else if constexpr (std::is_same<Scalar_t, half>::value) {
    return reinterpret_cast<half2 *>(ptr);
  } else {
    return ptr;
  }
}

template <class Vec_t> __host__ __device__ auto scalarize(void *ptr) {
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

template <class T> DEVICE_INLINE u16 get_col(T m) {
  return static_cast<u16>(m & T((1u << 16u) - 1u));
}

DEVICE_INLINE half get_val(u32 m) {
  u16 _v = m >> 16u;
  half v = *reinterpret_cast<half *>(&_v);
  return v;
}

#define CALL_FUSED(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH, IS_CSR) \
    constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT; \
    constexpr int BLOCK_WIDTH = _BLOCK_WIDTH; \
    size_t smem_size = sizeof(half2) * prob_n / 2;                   \
    F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR> \
            <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), \
            dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, __min(updiv(prob_m, 16), BLOCK_HEIGHT), 1), smem_size, \
            stream>>>(prob_m, \
            prob_n, \
            order_ptr, \
            raw_data_ptr,                               \
            X_ptr, \
            row_offsets_ptr, \
            col_vals_ptr, \
            y_ptr);


#define CALL_MATVEC(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH, IS_CSR) \
    constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT; \
    constexpr int BLOCK_WIDTH = _BLOCK_WIDTH; \
    size_t smem_size = sizeof(half2) * prob_n / 2;                   \
    F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR> \
            <<<dim3(updiv(prob_m, 16 * BLOCK_HEIGHT), 1, 1), \
            dim3(__min(updiv(prob_n, 16), BLOCK_WIDTH) * 16, __min(updiv(prob_m, 16), BLOCK_HEIGHT), 1), smem_size, \
            stream>>>(prob_m, \
            prob_n, \
            raw_data_ptr,                               \
            X_ptr, \
            row_offsets_ptr, \
            col_vals_ptr, \
            y_ptr);


static constexpr u32 SHARED_OFFSET = 32;

// Wait until at most `n` async copy stages are still pending.
template<int n> DEVICE_INLINE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n"::"n"(n));
}

DEVICE_INLINE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n");
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

constexpr int X_LOAD_BLOCK_SIZE = 8;
using Load_t = __int128_t;
constexpr bool PIPELINED_LOAD = false;


// #define DEQUANTIZE(v) make_half2(__int2half_rd((v) & 0b111), __int2half_rd(((v) >> 3) & 0b111))
#define INT2_TO_HALF2(v) s_half2_lut[v]

template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class W_t /* = uint64_t */, int
  PIPELINE_DEPTH, bool IS_CSR> __global__ void spqr_quantized_matvec_fused(
  // W and meta
  unsigned int prob_m,
  unsigned int prob_n,
  const uint16_t * __restrict__ in_order,
  // W 1st order stats
  const W_t *__restrict__ dense_matrix,
  const half *__restrict__ x,
  // Outliers
  const int *__restrict__ row_offsets,
  const u32 *__restrict__ col_vals,
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
  static constexpr u32 NUM_WARPS = UPDIV(NUM_HALF_WARPS, 2);
  static constexpr u32 THREAD_COUNT = BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;
  static constexpr u32 ROW_OFFSETS_SIZE = IS_CSR ? OUTPUT_SIZE : 1;

  constexpr int X_LOAD_BLOCK_SIZE = 2;

  extern __shared__ half2 s_x2[];
  static constexpr u32 LUT_SIZE = 64 * NUM_WARPS;
  __shared__ half2 s_half2_lut_global[LUT_SIZE];

  __shared__ u32 s_row_offsets[ROW_OFFSETS_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  if constexpr (THREAD_COUNT >= 64) {
    const auto v = make_half2(__int2half_rd(thread_xy & 0b111), __int2half_rd((thread_xy >> 3) & 0b111));
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) { s_half2_lut_global[i] = v; }
  } else {
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      const auto v = make_half2(__int2half_rd(i & 0b111u), __int2half_rd((i >> 3u) & 0b111u));
      s_half2_lut_global[i] = v;
    }
  }

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / WARP_SIZE) << 6);

  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

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

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS = 18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  float acc{};

  const auto* in_order_u32 = reinterpret_cast<const ushort2*>(in_order);

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= ROW_OFFSETS_SIZE; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i, row_offsets + blockIdx.x * ROW_OFFSETS_SIZE + i, sizeof(u32));
  }
  __pipeline_commit();

  __syncthreads();

  u32 i = subtile_id, pipeline_id{};
  const W_t *local_raw_data = dense_matrix + raw_data_offset;

  for (u32 x2_id = thread_xy, it = 0;
       it < UPDIV(prob_n / X_LOAD_BLOCK_SIZE, BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE); it++) {
    u32 idx = pipeline_id * THREAD_COUNT + thread_xy;

    RowBits row_bits{};
    half2 ws2{};
    half2 wz2{};

    bool p = idx < (prob_n / X_LOAD_BLOCK_SIZE);

    if (p) {
      auto in_order = in_order_u32[idx];
      s_x2[idx] = make_half2(x[in_order.x], x[in_order.y]);
    }

    auto v = __ldg(local_raw_data);
    row_bits.mask = v;
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    ws2 = __half2half2(first_order_dequantized.x);
    wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);
    __syncthreads();

#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = INT2_TO_HALF2(row_bits.get_w2(j + 1));
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }
    i += NUM_SPQR_TILES_PER_ITERATION;
    local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1;
    x2_id += NUM_SPQR_TILES_PER_ITERATION * BETA1;
    pipeline_id++;
  }

  for (; i < num_tiles_per_tile_row; i += NUM_SPQR_TILES_PER_ITERATION, local_raw_data +=
                                     NUM_SPQR_TILES_PER_ITERATION * BETA1) {
    auto v = __ldg(local_raw_data);
    RowBits row_bits{
      .mask = v
    };
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    half2 ws2 = __half2half2(first_order_dequantized.x);
    half2 wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);

#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = INT2_TO_HALF2(row_bits.get_w2(j + 1u));
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }
  }

  cp_async_wait<0>();

  if constexpr (IS_CSR) {
    u32 t = threadIdx.y * BETA1 + row_pos;
    u32 s = s_row_offsets[t];
    u32 e = s_row_offsets[t + 1];
    half *s_x = reinterpret_cast<half *>(s_x2);
    for (u32 i = s + subtile_id; i < e; i += BLOCK_WIDTH) {
      ColVal colval{
        ._ = __ldg(col_vals + i)
      };
      auto c = colval.members.c;
      auto v = colval.members.v;
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  } else {
    u32 s = s_row_offsets[0];
    u32 e = s_row_offsets[1];

    if (e - s) {
      half *s_x = reinterpret_cast<half *>(s_x2);

      if (s + thread_xy < e) {
        ColVal colval{._ = col_vals[s + thread_xy]};
        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }

      for (u32 i = s + thread_xy + BLOCK_WIDTH * BETA1; i < e; i += BLOCK_WIDTH * BETA1) {
        ColVal colval{
          ._ = col_vals[i]
        };

        if (!colval._) break;

        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }
    }
  }

  __syncthreads();
  auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
  acc = add_and_accum(other, acc);

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

  if (threadIdx.x < BETA1) {
    y_fp16[tile_row_id * BETA1 + threadIdx.x] = __float2half(acc);
  }
}

template<int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH, class W_t /* = uint64_t */, int
  PIPELINE_DEPTH, bool IS_CSR> __global__ void spqr_quantized_matvec(
  // W and meta
  unsigned int prob_m,
  unsigned int prob_n,
  // W 1st order stats
  const W_t *__restrict__ dense_matrix,
  const half *__restrict__ x,
  // Outliers
  const int *__restrict__ row_offsets,
  const u32 *__restrict__ col_vals,
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
  static constexpr u32 NUM_WARPS = UPDIV(NUM_HALF_WARPS, 2);
  static constexpr u32 THREAD_COUNT = BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;
  static constexpr u32 ROW_OFFSETS_SIZE = IS_CSR ? OUTPUT_SIZE : 1;

  extern __shared__ half2 s_x2[];
  static constexpr u32 LUT_SIZE = 64 * NUM_WARPS;
  __shared__ half2 s_half2_lut_global[LUT_SIZE];

  __shared__ u32 s_row_offsets[ROW_OFFSETS_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  if constexpr (THREAD_COUNT >= 64) {
    const auto v = make_half2(__int2half_rd(thread_xy & 0b111), __int2half_rd((thread_xy >> 3) & 0b111));
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) { s_half2_lut_global[i] = v; }
  } else {
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      const auto v = make_half2(__int2half_rd(i & 0b111u), __int2half_rd((i >> 3u) & 0b111u));
      s_half2_lut_global[i] = v;
    }
  }

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / WARP_SIZE) << 6);

  const half2 *x2 = reinterpret_cast<const half2 *>(x);


  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;


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

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS =
      18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  float acc{};

  __syncthreads();


  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= ROW_OFFSETS_SIZE; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i, row_offsets + blockIdx.x * ROW_OFFSETS_SIZE + i, sizeof(u32));
  }
  __pipeline_commit();


  u32 i = subtile_id, pipeline_id{};
  const W_t *local_raw_data = dense_matrix + raw_data_offset;


  for (u32 x2_id = thread_xy, it = 0;
       it < UPDIV(prob_n / X_LOAD_BLOCK_SIZE, BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE); it++) {
    u32 idx = pipeline_id * THREAD_COUNT + thread_xy;

    RowBits row_bits{};
    half2 ws2{};
    half2 wz2{};

    bool p = idx < (prob_n / X_LOAD_BLOCK_SIZE);

    if (p) {
      if constexpr (PIPELINED_LOAD) {
        auto x_global_load_ptr = reinterpret_cast<const Load_t *>(x2);
        auto x_shared_load_ptr = reinterpret_cast<const Load_t *>(s_x2);
        size_t smem_ptr = __cvta_generic_to_shared(x_shared_load_ptr + idx);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"l"(smem_ptr), "l"(x_global_load_ptr + idx));
      } else {
        reinterpret_cast<Load_t *>(s_x2)[idx] = reinterpret_cast<const Load_t *>(x2)[idx];
      }
    }

    if constexpr (PIPELINED_LOAD) {
      __pipeline_commit();
    }

    auto v = __ldg(local_raw_data);
    row_bits.mask = v;
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    ws2 = __half2half2(first_order_dequantized.x);
    wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);
    if constexpr (PIPELINED_LOAD) {
      cp_async_wait_all();
    }
    __syncthreads();

#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = INT2_TO_HALF2(row_bits.get_w2(j + 1));
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }
    i += NUM_SPQR_TILES_PER_ITERATION;
    local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1;
    x2_id += NUM_SPQR_TILES_PER_ITERATION * BETA1;
    pipeline_id++;
  }

  for (; i < num_tiles_per_tile_row; i += NUM_SPQR_TILES_PER_ITERATION, local_raw_data +=
                                     NUM_SPQR_TILES_PER_ITERATION * BETA1) {
    auto v = __ldg(local_raw_data);
    RowBits row_bits{
      .mask = v
    };
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS) << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized = dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

    half2 ws2 = __half2half2(first_order_dequantized.x);
    half2 wz2 = __half2half2(first_order_dequantized.y);

    const auto s_x2_ = s_x2 + i * (BETA2 >> 1);

#pragma unroll
    for (u32 j = 0; j < BETA2 / 2; j++) {
      half2 w_q = INT2_TO_HALF2(row_bits.get_w2(j + 1u));
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(s_x2_[j]);
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }
  }

  cp_async_wait_all();
  if constexpr (IS_CSR) {
    u32 t = threadIdx.y * BETA1 + row_pos;
    u32 s = s_row_offsets[t];
    u32 e = s_row_offsets[t + 1];
    half *s_x = reinterpret_cast<half *>(s_x2);
    for (u32 i = s + subtile_id; i < e; i += BLOCK_WIDTH) {
      ColVal colval{
        ._ = __ldg(col_vals + i)
      };
      auto c = colval.members.c;
      auto v = colval.members.v;
      acc += __half2float(v) * __half2float(s_x[c]);
    }
  } else {
    u32 s = s_row_offsets[0];
    u32 e = s_row_offsets[1];

    if (e - s) {
      half *s_x = reinterpret_cast<half *>(s_x2);

      if (s + thread_xy < e) {
        ColVal colval{._ = col_vals[s + thread_xy]};
        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }

      for (u32 i = s + thread_xy + BLOCK_WIDTH * BETA1; i < e; i += BLOCK_WIDTH * BETA1) {
        ColVal colval{
          ._ = col_vals[i]
        };

        if (!colval._) break;

        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }
    }
  }



  __syncthreads();
  auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
  acc = add_and_accum(other, acc);

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

  if (threadIdx.x < BETA1) {
    y_fp16[tile_row_id * BETA1 + threadIdx.x] = __float2half(acc);
  }
}

template<class T> const T &__min(const T &a, const T &b) {
  return (b < a) ? b : a;
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
  const void *raw_in_order,
  const void *raw_dense_data,
  // 32-bit
  int row_offsets_len,
  void *row_offsets,
  // 16-bit
  void *col_vals,
  int nnz,
  // 16-bit
  // Input
  void *X,
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

  const auto *raw_data_ptr = (const u64 *) raw_dense_data;
  const half *X_ptr = (const half *) X;
  const int *row_offsets_ptr = (const int *) row_offsets;
  half *y_ptr = (half *) y;
  const auto *col_vals_ptr = (const u32 *) col_vals;
  const auto *order_ptr = (const uint16_t *) raw_in_order;

  int ret = 0;

  bool is_csr = prob_m + 1 == row_offsets_len;


  if (order_ptr == nullptr) {
    if (is_csr) {
      if (prob_m % 16 == 0 && prob_n % 512 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 256 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 8, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 4, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 2, 1, true);
      } else {
        CALL_MATVEC(spqr_quantized_matvec, 1, 1, 1, true);
      }
    } else {
      if (prob_m % 16 == 0 && prob_n % 512 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 256 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 8, 2, false);
      } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 4, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 2, 1, false);
      } else {
        CALL_MATVEC(spqr_quantized_matvec, 1, 1, 1, false);
      }
    }
  } else { 
    if (is_csr) {
      if (prob_m % 16 == 0 && prob_n % 512 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 256 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 8, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 4, 1, true);
      } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 2, 1, true);
      } else {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 1, 1, true);
      }
    } else {
      if (prob_m % 16 == 0 && prob_n % 512 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 256 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 128 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 8, 2, false);
      } else if (prob_m % 16 == 0 && prob_n % 64 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 4, 1, false);
      } else if (prob_m % 16 == 0 && prob_n % 32 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 2, 1, false);
      } else {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 1, 1, false);
      }
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
