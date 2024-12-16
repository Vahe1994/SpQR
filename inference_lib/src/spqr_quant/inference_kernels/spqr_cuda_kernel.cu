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

#include <cassert>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define DEVICE_INLINE __forceinline__ __device__

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

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

#define INT2_TO_HALF2(v)                                                       \
  make_half2(__int2half_rd((v) & 0b111), __int2half_rd(((v) >> 3) & 0b111))

template <class Acc_t> constexpr __device__ __host__ bool is_fp32() {
  if constexpr (std::is_same_v<Acc_t, float> || std::is_same_v<Acc_t, float2>) {
    return true;
  }
  return false;
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut> __device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n> struct Vec {
  T elems[n];
  DEVICE_INLINE T &operator[](int i) { return elems[i]; }
  DEVICE_INLINE const T operator[](int i) const { return elems[i]; }
};

using I4 = Vec<int, 4>;

using Frag4 = Vec<half2, 2>;

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ __forceinline__ Frag4 dequant(int q) {
  q = (q & 0b111) | ((q & 0b111000000) >> 2) | ((q & 0b111000) << 13) |
      ((q & 0b111000000000) << 11);

  // q = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  // q = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  // q |= 1000;

  // q = (q & 0b111) |
  // ((q & 0b111000) << 1) |
  //   ((q & 0b111000000) << 2) |
  //   ((q & 0b111000000000) << 3);

  // q = (q & 0b111) |
  //   ((q & 0b111000000) >> 2) |
  //   ((q & 0b111000) << 5) |
  //     ((q & 0b111000000000) << 3);

  // q = (q & 0b111) |
  // ((q & 0b111000) << 1) |
  //   ((q & 0b111000000) << 2) |
  //   ((q & 0b111000000000) << 3);

  // q |= 0b1000100010001000;

  constexpr int LO = 0x000f000f;
  constexpr int HI = 0x00f000f0;
  constexpr int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3 < (0xf0 & 0xcc) | 0xaa > (q, LO, EX);
  int hi = lop3 < (0xf0 & 0xcc) | 0xaa > (q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  constexpr int SUB = 0x64006400;
  constexpr int MUL = 0x2c002c00;
  constexpr int ADD = 0xd400d400;
  Frag4 frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                      *reinterpret_cast<const half2 *>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2 *>(&hi),
                      *reinterpret_cast<const half2 *>(&MUL),
                      *reinterpret_cast<const half2 *>(&ADD));
  return frag_b;
}

DEVICE_INLINE Vec<half2, 2> transpose_2x2(const Vec<half2, 2> &a) {
  return Vec<half2, 2>{
      .elems = {make_half2(a[0].x, a[1].x), make_half2(a[0].y, a[1].y)}};
}

DEVICE_INLINE Vec<half2, 2> transpose_4x4(const Vec<half2, 2> &a) {
  return Vec<half2, 2>{
      .elems = {make_half2(a[0].x, a[1].x), make_half2(a[0].y, a[1].y)}};
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ __forceinline__ half2 dequant2(int q) {
  constexpr int EX = 0x64006400;

  int lo = (q & 0b111) | (((q & 0b111000) << 13)) | EX;
  constexpr int SUB = 0x64006400;
  return __hsub2(*reinterpret_cast<half2 *>(&lo),
                 *reinterpret_cast<const half2 *>(&SUB));
}

DEVICE_INLINE uint64_t recover_second_order_sync(uint64_t val) {
  static constexpr unsigned int FULL_MASK = 0xFFFFFFFFu;
  val |= __shfl_xor_sync(FULL_MASK, val, 2);
  val |= __shfl_xor_sync(FULL_MASK, val, 4);
  val |= __shfl_xor_sync(FULL_MASK, val, 8);
  return val;
}

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

DEVICE_INLINE float add_and_accum(float a, float b) { return a + b; }

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

#define CALL_FUSED(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH, IS_CSR)     \
  constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT;                                  \
  constexpr int BLOCK_WIDTH = _BLOCK_WIDTH;                                    \
  size_t smem_size = __max(4096 * sizeof(half2), sizeof(half2) * n / 2);       \
  F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR>         \
      <<<dim3(updiv(m, 16 * BLOCK_HEIGHT), 1, 1),                              \
         dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16,                           \
              __min(updiv(m, 16), BLOCK_HEIGHT), 1),                           \
         smem_size, stream>>>(m, n, order_ptr, raw_data_ptr, X_ptr,            \
                              row_offsets_ptr, col_vals_ptr, y_ptr);

#define CALL_MATVEC(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH, IS_CSR)    \
  constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT;                                  \
  constexpr int BLOCK_WIDTH = _BLOCK_WIDTH;                                    \
  size_t smem_size = __max(4096 * sizeof(half2), sizeof(half2) * n / 2);       \
  F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR>         \
      <<<dim3(updiv(m, 16 * BLOCK_HEIGHT), 1, 1),                              \
         dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16,                           \
              __min(updiv(m, 16), BLOCK_HEIGHT), 1),                           \
         smem_size, stream>>>(m, n, raw_data_ptr, X_ptr, row_offsets_ptr,      \
                              col_vals_ptr, y_ptr);

#define CALL_BATCHED(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH, IS_CSR,   \
                     K)                                                        \
  constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT;                                  \
  constexpr int BLOCK_WIDTH = _BLOCK_WIDTH;                                    \
  size_t smem_size = max(4096 * sizeof(half2), sizeof(half2) * (n / 2) * K);   \
  F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR, K>      \
      <<<dim3(updiv(m, 16 * BLOCK_HEIGHT), 1, 1),                              \
         dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16,                           \
              __min(updiv(m, 16), BLOCK_HEIGHT), 1),                           \
         smem_size, stream>>>(m, n, raw_data_ptr, X_ptr, row_offsets_ptr,      \
                              col_vals_ptr, y_ptr);

#define CALL_BATCHED_V2(F, _BLOCK_HEIGHT, _BLOCK_WIDTH, PIPELINE_DEPTH,        \
                        IS_CSR, K)                                             \
  constexpr int BLOCK_HEIGHT = _BLOCK_HEIGHT;                                  \
  constexpr int BLOCK_WIDTH = _BLOCK_WIDTH;                                    \
  constexpr int page_size_fp32 = 4096;                                         \
  F<3, 16, 16, BLOCK_HEIGHT, BLOCK_WIDTH, u64, PIPELINE_DEPTH, IS_CSR, K,      \
    page_size_fp32><<<dim3(updiv(m, 16 * BLOCK_HEIGHT), 1, 1),                 \
                      dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16,              \
                           __min(updiv(m, 16), BLOCK_HEIGHT), 1),              \
                      page_size_fp32 * sizeof(float), stream>>>(               \
      m, n, raw_data_ptr, X_ptr, row_offsets_ptr, col_vals_ptr, y_ptr);

static constexpr u32 SHARED_OFFSET = 32;

// Wait until at most `n` async copy stages are still pending.
template <int n> DEVICE_INLINE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

DEVICE_INLINE void cp_async(half2 *__restrict__ dst,
                            const half2 *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s_dst),
               "l"(src));
}

using Load_t = __int128_t;
DEVICE_INLINE void cp_async128(Load_t *__restrict__ dst,
                            const Load_t *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s_dst),
               "l"(src));
}

DEVICE_INLINE void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

__device__ __forceinline__ uint32_t __ld_stream(const uint32_t *ptr) {
  uint32_t v;
  asm volatile("{\n"
               "   ld.global.ca.u32 %0, [%1];\n"
               "}\n"
               : "=r"(v)
               : "l"(ptr));
  return v;
}

constexpr int X_LOAD_BLOCK_SIZE = 8;
constexpr bool PIPELINED_LOAD = false;

// #define INT2_TO_HALF2(v) s_half2_lut[v]
// #define INT2_TO_HALF2(v) s_half2_lut[(thread_xy + v) & 0x2F]

template <int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          class W_t /* = uint64_t */, int PIPELINE_DEPTH, bool IS_CSR>
__global__ void spqr_quantized_matvec_fused(
    // W and meta
    unsigned int m, unsigned int n, const uint16_t *__restrict__ in_order,
    // W 1st order stats
    const W_t *__restrict__ dense_matrix, const half *__restrict__ x,
    // Outliers
    const int *__restrict__ row_offsets, const u32 *__restrict__ col_vals,
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
  static constexpr u32 THREAD_COUNT =
      BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;
  static constexpr u32 ROW_OFFSETS_SIZE = IS_CSR ? OUTPUT_SIZE : 1;

  constexpr int X_LOAD_BLOCK_SIZE = 2;

  extern __shared__ half2 s_x2[];
  static constexpr u32 LUT_SIZE = 64 * NUM_WARPS;
  __shared__ half2 s_half2_lut_global[LUT_SIZE];

  __shared__ u32 s_row_offsets[ROW_OFFSETS_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  if constexpr (THREAD_COUNT >= 64) {
    const auto v = make_half2(__int2half_rd(thread_xy & 0b111),
                              __int2half_rd((thread_xy >> 3) & 0b111));
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      s_half2_lut_global[i] = v;
    }
  } else {
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      const auto v = make_half2(__int2half_rd(i & 0b111u),
                                __int2half_rd((i >> 3u) & 0b111u));
      s_half2_lut_global[i] = v;
    }
  }

  auto s_half2_lut = s_half2_lut_global + ((thread_xy / WARP_SIZE) << 6);

  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_tiles_per_tile_row = UPDIV(n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  u32 row_pos = thread_xy & 0xF;
  const u32 subtile_id = threadIdx.x / BETA1;

  auto raw_data_offset = tile_row_id * n + threadIdx.x;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS =
      18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  float acc{};

  const auto *in_order_u32 = reinterpret_cast<const ushort2 *>(in_order);

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= ROW_OFFSETS_SIZE; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i,
                            row_offsets + blockIdx.x * ROW_OFFSETS_SIZE + i,
                            sizeof(u32));
  }
  __pipeline_commit();

  __syncthreads();

  u32 i = subtile_id, pipeline_id{};
  const W_t *local_raw_data = dense_matrix + raw_data_offset;

  for (u32 x2_id = thread_xy, it = 0;
       it < UPDIV(n / X_LOAD_BLOCK_SIZE,
                  BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE);
       it++) {
    u32 idx = pipeline_id * THREAD_COUNT + thread_xy;

    RowBits row_bits{};
    half2 ws2{};
    half2 wz2{};

    bool p = idx < (n / X_LOAD_BLOCK_SIZE);

    if (p) {
      auto in_order = in_order_u32[idx];
      s_x2[idx] = make_half2(x[in_order.x], x[in_order.y]);
    }

    auto v = __ldg(local_raw_data);
    row_bits.mask = v;
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS)
        << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized =
        dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

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

  for (; i < num_tiles_per_tile_row;
       i += NUM_SPQR_TILES_PER_ITERATION,
       local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1) {
    auto v = __ldg(local_raw_data);
    RowBits row_bits{.mask = v};
    uint64_t s_order_partial =
        (row_bits.mask >> NUM_USEFUL_BITS)
        << (SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET));
    SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};
    half2 first_order_quantized = INT2_TO_HALF2(row_bits.get_w2(0));
    half2 first_order_dequantized =
        dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

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
      ColVal colval{._ = __ldg(col_vals + i)};
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

      for (u32 i = s + thread_xy + BLOCK_WIDTH * BETA1; i < e;
           i += BLOCK_WIDTH * BETA1) {
        ColVal colval{._ = col_vals[i]};

        if (!colval._)
          break;

        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }
    }
  }

  __syncthreads();
  auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
  acc = add_and_accum(other, acc);

  auto *s_fp32_buff = reinterpret_cast<float *>(
      s_half2_lut_global + threadIdx.y * MAX(WARP_SIZE - 1, 1) * BETA1);

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

__device__ __forceinline__ float accumulate(float acc, u64 b, const half2 &ws2,
                                            const half2 &wz2,
                                            const half2 *__restrict__ s_x2) {
#if 0
#pragma unroll
  for (u32 j = 0; j < 8; j++) {
    b >>= 6u;
    half2 w_q = INT2_TO_HALF2(b & 0b111111ull);
    half2 w = dequantize2(w_q, ws2, wz2);
    float2 x_fp32 = __half22float2(s_x2[j]);
    float2 w_fp32 = __half22float2(w);
    acc = fmaf(x_fp32.x, w_fp32.x, acc);
    acc = fmaf(x_fp32.y, w_fp32.y, acc);
  }
  return acc;
#else
  b >>= 6u;
#pragma unroll
  for (u32 i = 0; i < 4; i++) {
    auto frag = dequant(b);

#pragma unroll
    for (u32 j = 0; j < 2; j++) {
      auto w_q = frag.elems[j];
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 x_fp32 = __half22float2(*(s_x2++));
      float2 w_fp32 = __half22float2(w);
      acc = fmaf(x_fp32.x, w_fp32.x, acc);
      acc = fmaf(x_fp32.y, w_fp32.y, acc);
    }
    b >>= 12ull;
  }
  return acc;
#endif
}

template <int K>
__device__ __forceinline__ Vec<float, K>
accumulate_batched(Vec<float, K> acc, u64 b, const half2 &ws2, const half2 &wz2,
                   const half2 *__restrict__ s_x2) {
  b >>= 6u;
#pragma unroll
  for (u32 i = 0; i < 4; i++) {
    auto frag = dequant(b);

#pragma unroll
    for (int j = 0; j < 2; j++) {
      auto w_q = frag.elems[j];
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 w_fp32 = __half22float2(w);

      if constexpr (K == 1) {
        // printf("%f %f\n", __half2float(s_x2[0].x), __half2float(s_x2[1].y));
        float2 x_fp32 = __half22float2(*(s_x2++));
        acc[0] = fmaf(x_fp32.x, w_fp32.x, acc[0]);
        acc[0] = fmaf(x_fp32.y, w_fp32.y, acc[0]);
      } else {
#pragma loop unroll
        for (int l = 0; l < K / 2; l++) {
          Vec<half2, 2> x{.elems = {s_x2[l], s_x2[l + K / 2]}};
          Vec<half2, 2> x_t = transpose_2x2(x);

#pragma loop unroll
          for (int k = 0; k < 2; k++) {
            float2 x_fp32 = __half22float2(x_t[k]);
            acc[2 * l + k] = fmaf(x_fp32.x, w_fp32.x, acc[2 * l + k]);
            acc[2 * l + k] = fmaf(x_fp32.y, w_fp32.y, acc[2 * l + k]);
          }
        }
        s_x2 += K;
      }
    }
    b >>= 12ull;
  }
  return acc;
}

template <int K>
__device__ __forceinline__ Vec<float, K>
accumulate_batched_lut(Vec<float, K> acc, u64 b, const half2 &ws2,
                       const half2 &wz2, const half2 *__restrict__ s_x2,
                       const half2 *__restrict__ lut) {
  b >>= 6u;
#pragma unroll
  for (u32 i = 0; i < 4; i++) {
    //    auto frag = dequant(b);
    Frag4 frag{lut[b & 0b111111u], lut[(b & 0b111111000000) >> 6]};

#pragma unroll
    for (int j = 0; j < 2; j++) {
      auto w_q = frag.elems[j];
      half2 w = dequantize2(w_q, ws2, wz2);
      float2 w_fp32 = __half22float2(w);

      if constexpr (K == 1) {
        float2 x_fp32 = __half22float2(*(s_x2++));
        acc[0] = fmaf(x_fp32.x, w_fp32.x, acc[0]);
        acc[0] = fmaf(x_fp32.y, w_fp32.y, acc[0]);
      } else {
#pragma loop unroll
        for (int l = 0; l < K / 2; l++) {
          Vec<half2, 2> x{.elems = {s_x2[l], s_x2[l + K / 2]}};
          Vec<half2, 2> x_t = transpose_2x2(x);

#pragma loop unroll
          for (int k = 0; k < 2; k++) {
            float2 x_fp32 = __half22float2(x_t[k]);
            acc[2 * l + k] = fmaf(x_fp32.x, w_fp32.x, acc[2 * l + k]);
            acc[2 * l + k] = fmaf(x_fp32.y, w_fp32.y, acc[2 * l + k]);
          }
        }
        s_x2 += K;
      }
    }
    b >>= 12ull;
  }
  return acc;
}

template <class W_t, int X_LOAD_BLOCK_SIZE, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          int HALF_WARP_SIZE, int THREAD_COUNT, int NUM_USEFUL_BITS, int OFFSET,
          int BETA1, int BETA2, int NUM_SPQR_TILES_PER_ITERATION>
struct DenseMatrixRunner {
  u32 i;
  const W_t *__restrict local_raw_data;
  u32 thread_xy;
  u32 n;
  const half2 *__restrict__ x2;
  half2 *s_x2;
  u32 row_pos;
  u32 num_tiles_per_row;
  u32 subtile_id;

  float acc{};
  u32 pipeline_id{};
  u32 it{};

  template <bool LOAD_X> __device__ __forceinline__ void process_dense() {
    const uint64_t SHIFT = SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET);
    const u32 BLOCK_COUNT = (n / X_LOAD_BLOCK_SIZE);

    u32 limit;
    if constexpr (LOAD_X) {
      limit = UPDIV(n / X_LOAD_BLOCK_SIZE,
                    BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE);
    } else {
      limit = num_tiles_per_row / NUM_SPQR_TILES_PER_ITERATION;
    }
    for (; it < limit; it++) {
      if constexpr (LOAD_X) {
        u32 idx = pipeline_id * THREAD_COUNT + thread_xy;
        bool p = idx < BLOCK_COUNT;
        if (p) {
          reinterpret_cast<Load_t *>(s_x2)[idx] =
              reinterpret_cast<const Load_t *>(x2)[idx];
        }
      }

      auto v = __ldg(local_raw_data);
      uint64_t s_order_partial = (v >> NUM_USEFUL_BITS) << SHIFT;

      SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};

      half2 first_order_quantized = dequant2(v);

      half2 first_order_dequantized =
          dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

      half2 ws2 = __half2half2(first_order_dequantized.x);
      half2 wz2 = __half2half2(first_order_dequantized.y);

      const auto s_x2_ = s_x2 + i * (BETA2 >> 1);
      if constexpr (LOAD_X) {
        __syncthreads();
      }

      acc = accumulate(acc, v, ws2, wz2, s_x2_);

      i += NUM_SPQR_TILES_PER_ITERATION;
      local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1;
      pipeline_id++;
    }
  }
};

template <class W_t, int X_LOAD_BLOCK_SIZE, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          int HALF_WARP_SIZE, int THREAD_COUNT, int NUM_USEFUL_BITS, int OFFSET,
          int BETA1, int BETA2, int NUM_SPQR_TILES_PER_ITERATION, int K,
          int page_size_fp32>
struct DenseMatrixRunnerBatched {
  const W_t *__restrict local_raw_data;
  u32 thread_xy;
  u32 n;
  const half2 *__restrict__ x2;
  half2 *__restrict__ s_x2;
  u32 row_pos;
  u32 subtile_id;
  half2 *__restrict__ lut;

  Vec<float, K> accs{};
  u32 global_x_fp128_loaded_base_id;
  int global_x_fp16_computed_count;

  DEVICE_INLINE void init() {
    global_x_fp128_loaded_base_id = 0;
    global_x_fp16_computed_count = subtile_id * BETA1 * K;
  }

  DEVICE_INLINE void process_dense() {
    const uint64_t SHIFT = SECOND_ORDER_FRAGMENT_SIZE_BITS * (row_pos / OFFSET);

    // For higher batch sizes, we don't have any guarantees that smem will be
    // large enough to fit x.

    // In each iteration, in order to complete the block-wise multiplication,
    // we need to load k * beta1 * num_blocks FP16 weights.
    //
    //           k
    //        ┌─────┐
    //        │     │
    //        │     │
    //        │ X_i │ BETA1 * BLOCK_WIDTHS
    //        │     │
    //        │     │
    //        └─────┘

    // We will end up loading this many FP16s while reading X during a single
    // iteration.
    const int total_x_fp32 = n * K / 2;
    static constexpr int total_x_fp16_per_iteration = BETA1 * BLOCK_WIDTH * K;
    static constexpr int total_x_fp32_load_per_iteration =
        total_x_fp16_per_iteration / 2;
    static constexpr int total_x_fp128_per_iteration =
        total_x_fp32_load_per_iteration / 4;

#if 0
    if (!thread_xy && !blockIdx.x)
      printf("smem_size = %d\n", page_size_fp32);
#endif
    //           k
    //        ┌─────┐        ┬
    //        │     │        │
    //        ├─────┤        │
    //        │     │        │
    //        ├─────┤        │
    //        │ ... │        n
    //        ├─────┤        │
    //        │     │        │
    //        ├─────┤        │
    //        │     │        │
    //        └─────┘        ┴

    // However, we can't fit the entire input matrix into shared memory.
    // We can only store smem_size FP32s at any given time. Therefore,
    // we have to page the loads into X.
    //           k
    //        ┌─────┐               ┬     ┬
    //        │     │               │     │
    //        │     │   fp32s_per_page    │
    //        │     │               │     │
    //        ├─────┤               ┴     │
    //        │     │               │     │
    //        │     │   fp32s_per_page    │
    //        │     │               │     │
    //        └─────┘               ┴     ┴

    // Therefore, we will page the loads of X, loading fp32s_per_page
    // per iteration.

    auto s_x2_load = s_x2;
    auto s_x2_compute = s_x2 + subtile_id * (BETA2 * K / 2);

    int local_x_fp128_loaded_base_id{};


    auto s_x128 = reinterpret_cast<Load_t*>(s_x2);
    const auto x128 = reinterpret_cast<const Load_t*>(x2);

    for (;;) {
      for (int i = thread_xy;
           i < total_x_fp32_load_per_iteration / 4 &&
           local_x_fp128_loaded_base_id + i < page_size_fp32 / 4 &&
           global_x_fp128_loaded_base_id + i < K * n / 8;
           i += THREAD_COUNT) {
        cp_async128(s_x128 + i, x128 + global_x_fp128_loaded_base_id + i);
        // s_x2_load[i] = x2[global_x_fp32_loaded_base_id + i];
      }
      __pipeline_commit();

      // Streaming data - will only be used once and only once
      // by a single thread
      // auto v = __ldcs(local_raw_data);
      auto v = __ldcs(local_raw_data);
      uint64_t s_order_partial = (v >> NUM_USEFUL_BITS) << SHIFT;

      SecondOrder _s{.v = recover_second_order_sync(s_order_partial)};

      half2 first_order_quantized = dequant2(v);

      half2 first_order_dequantized =
          dequantize2(first_order_quantized, _s.get_sws2(), _s.get_swz2());

      half2 ws2 = __half2half2(first_order_dequantized.x);
      half2 wz2 = __half2half2(first_order_dequantized.y);

      cp_async_wait<0>();
      __syncthreads();

      accs = accumulate_batched_lut<K>(accs, v, ws2, wz2, s_x2_compute, lut);

      local_raw_data += NUM_SPQR_TILES_PER_ITERATION * BETA1;

      s_x2_compute += BETA2 * K * BLOCK_WIDTH / 2;
      s_x128 += total_x_fp32_load_per_iteration / 4;

      local_x_fp128_loaded_base_id += total_x_fp32_load_per_iteration / 4;

      global_x_fp128_loaded_base_id += total_x_fp32_load_per_iteration / 4;

      bool global_load_pred = global_x_fp128_loaded_base_id < total_x_fp32 / 4 &&
                              local_x_fp128_loaded_base_id < page_size_fp32 / 4;

      if (!global_load_pred) {
        break;
      }
    }
  }
};

template <int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          class W_t /* = uint64_t */, int PIPELINE_DEPTH, bool IS_CSR>
__global__ void spqr_quantized_matvec(
    // W and meta
    unsigned int m, unsigned int n,
    // W 1st order stats
    const W_t *__restrict__ dense_matrix, const half *__restrict__ x,
    // Outliers
    const int *__restrict__ row_offsets, const u32 *__restrict__ col_vals,
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
  static constexpr u32 THREAD_COUNT =
      BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;
  static constexpr u32 ROW_OFFSETS_SIZE = IS_CSR ? OUTPUT_SIZE : 1;

  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[ROW_OFFSETS_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  const half2 *x2 = reinterpret_cast<const half2 *>(x);

  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_tiles_per_tile_row = UPDIV(n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  u32 row_pos = thread_xy & 0xF;
  const u32 subtile_id = threadIdx.x / BETA1;

  auto raw_data_offset = tile_row_id * n + threadIdx.x;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS =
      18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  __syncthreads();

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= ROW_OFFSETS_SIZE; i += THREAD_COUNT) {
    __pipeline_memcpy_async(s_row_offsets + i,
                            row_offsets + blockIdx.x * ROW_OFFSETS_SIZE + i,
                            sizeof(u32));
  }
  __pipeline_commit();

  DenseMatrixRunner<W_t, X_LOAD_BLOCK_SIZE, BLOCK_HEIGHT, BLOCK_WIDTH,
                    HALF_WARP_SIZE, THREAD_COUNT, NUM_USEFUL_BITS, OFFSET,
                    BETA1, BETA2, NUM_SPQR_TILES_PER_ITERATION>
      dense_matrix_runner{.i = subtile_id,
                          .local_raw_data = dense_matrix + raw_data_offset,
                          .thread_xy = thread_xy,
                          .n = n,
                          .x2 = x2,
                          .s_x2 = s_x2,
                          .row_pos = row_pos,
                          .num_tiles_per_row = num_tiles_per_tile_row,
                          .subtile_id = subtile_id};

  dense_matrix_runner.template process_dense<true>();

  dense_matrix_runner.template process_dense<false>();

  float acc = dense_matrix_runner.acc;

  cp_async_wait_all();
  if constexpr (IS_CSR) {
    u32 t = threadIdx.y * BETA1 + row_pos;
    u32 s = s_row_offsets[t];
    u32 e = s_row_offsets[t + 1];
    half *s_x = reinterpret_cast<half *>(s_x2);
    for (u32 i = s + subtile_id; i < e; i += BLOCK_WIDTH) {
      ColVal colval{._ = __ldg(col_vals + i)};
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

      for (u32 i = s + thread_xy + BLOCK_WIDTH * BETA1; i < e;
           i += BLOCK_WIDTH * BETA1) {
        ColVal colval{._ = col_vals[i]};

        if (!colval._)
          break;

        auto c = colval.members.c;
        auto v = colval.members.v;
        acc += __half2float(v) * __half2float(s_x[c]);
      }
    }
  }

  __syncthreads();

  auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
  acc = add_and_accum(other, acc);

  // TODO: Invalid read if x is not large enough
  auto *s_fp32_buff = reinterpret_cast<float *>(
      s_x2 + threadIdx.y * MAX(WARP_SIZE - 1, 1) * BETA1);

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

template <int BITS, int BETA1, int BETA2, int BLOCK_HEIGHT, int BLOCK_WIDTH,
          class W_t /* = uint64_t */, int PIPELINE_DEPTH, bool IS_CSR, int K,
          int page_size_fp32>
__global__ void spqr_quantized_matvec_batched_v2(
    // W and meta
    u32 m, u32 n,
    // W 1st order stats
    const W_t *__restrict__ dense_matrix, const half *__restrict__ x,
    // Outliers
    const int *__restrict__ row_offsets, const u32 *__restrict__ col_vals,
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
  static constexpr u32 THREAD_COUNT =
      BLOCK_HEIGHT * BLOCK_WIDTH * HALF_WARP_SIZE;
  static constexpr u32 OUTPUT_SIZE = BETA1 * BLOCK_HEIGHT;
  static constexpr u32 ROW_OFFSETS_SIZE = IS_CSR ? OUTPUT_SIZE : 1;

  half2 *y_fp32 = reinterpret_cast<half2 *>(y_fp16);
  extern __shared__ half2 s_x2[];
  __shared__ u32 s_row_offsets[ROW_OFFSETS_SIZE + 1];

  const u32 thread_xy = threadIdx.x + (threadIdx.y * blockDim.x);

  const half2 *x2 = reinterpret_cast<const half2 *>(x);

  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  // Number of SPQR tiles that this CUDA block will process.
  u32 num_tiles_per_tile_row = UPDIV(n, BETA2);

  // Here is how we organize things here. We have THREAD_COUNT threads in a
  // block in x-dimension. We distribute 1 thread per tile row. Therefore, we
  // have BETA1 threads per tile. For now, a block only spans across 1 dimension
  // of SPQR tiles.
  constexpr u32 NUM_SPQR_TILES_PER_ITERATION = BLOCK_WIDTH;
  constexpr u32 WARP_COUNT = UPDIV(BLOCK_WIDTH, 2);

  u32 row_pos = thread_xy & 0xF;
  const u32 subtile_id = threadIdx.x / BETA1;

  auto raw_data_offset = tile_row_id * n + threadIdx.x;

  constexpr u32 FULL_MASK = 0xffffffff;
  constexpr u32 HALF_MASK = FULL_MASK >> 16u;

  constexpr static unsigned long long int NUM_USEFUL_BITS =
      18ull * static_cast<u64>(BITS);
  constexpr static int OFFSET = BETA1 / SECOND_ORDER_FRAGMENT_SIZE_BITS;

  static constexpr u32 LUT_SIZE = 64;
  __shared__ half2 lut[LUT_SIZE];
  if constexpr (THREAD_COUNT >= 64) {
    const auto v = make_half2(__int2half_rd(thread_xy & 0b111),
                              __int2half_rd((thread_xy >> 3) & 0b111));
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      lut[i] = v;
    }
  } else {
#pragma unroll
    for (u32 i = thread_xy; i < LUT_SIZE; i += THREAD_COUNT) {
      const auto v = make_half2(__int2half_rd(i & 0b111u),
                                __int2half_rd((i >> 3u) & 0b111u));
      lut[i] = v;
    }
  }

  // Here we load the row offsets into smem.
  for (u32 i = thread_xy; i <= ROW_OFFSETS_SIZE; i += THREAD_COUNT) {
    s_row_offsets[i] = row_offsets[blockIdx.x * ROW_OFFSETS_SIZE + i];
  }

  DenseMatrixRunnerBatched<W_t, X_LOAD_BLOCK_SIZE, BLOCK_HEIGHT, BLOCK_WIDTH,
                           HALF_WARP_SIZE, THREAD_COUNT, NUM_USEFUL_BITS,
                           OFFSET, BETA1, BETA2, NUM_SPQR_TILES_PER_ITERATION,
                           K, page_size_fp32>
      dense_matrix_runner{.local_raw_data = dense_matrix + raw_data_offset,
                          .thread_xy = thread_xy,
                          .n = n,
                          .x2 = x2,
                          .s_x2 = s_x2,
                          .row_pos = row_pos,
                          .subtile_id = subtile_id,
                          .lut = lut};

  dense_matrix_runner.init();

  __syncthreads();

  u32 t = threadIdx.y * BETA1 + row_pos;
  u32 s, e, i;
  if constexpr (IS_CSR) {
    s = s_row_offsets[t];
    e = s_row_offsets[t + 1];
    i = s + subtile_id;
  } else {
    s = s_row_offsets[0];
    e = s_row_offsets[1];
    i = s + thread_xy;
  }

  const int total_x_fp32 = n * K / 2;
  int pipeline_stages = UPDIV(total_x_fp32, page_size_fp32);

  for (int pipeline_id{}; pipeline_id < pipeline_stages; pipeline_id++) {
    dense_matrix_runner.process_dense();

    if constexpr (IS_CSR) {
      for (; i < e; i += BLOCK_WIDTH) {
        ColVal colval{._ = __ldg(col_vals + i)};
        auto c = colval.members.c;

        if (c * K / 2 >= (pipeline_id + 1) * page_size_fp32) {
          break;
        }

        auto v = colval.members.v;
        float v_fp32 = __half2float(v);

        if constexpr (K == 1) {
          half *s_x = reinterpret_cast<half *>(s_x2);
          int idx = c - pipeline_id * page_size_fp32 * 2;
          dense_matrix_runner.accs[0] +=
              __half2float(v) * __half2float(s_x[idx]);
        } else {
#pragma loop unroll
          for (int j = 0; j < K; j += 2) {
            int idx = c * K / 2 + j / 2 - pipeline_id * page_size_fp32;
            float2 x2_fp32 = __half22float2(s_x2[idx]);
            dense_matrix_runner.accs[j] += v_fp32 * x2_fp32.x;
            dense_matrix_runner.accs[j + 1] += v_fp32 * x2_fp32.y;
          }
        }
      }
    } else {
      for (; i < e; i += BLOCK_WIDTH * BETA1) {
        ColVal colval{._ = col_vals[i]};

        if (!colval._) {
          break;
        }

        auto c = colval.members.c;

        if (c * K / 2 >= (pipeline_id + 1) * page_size_fp32) {
          break;
        }

        auto v = colval.members.v;
        float v_fp32 = __half2float(v);

        if constexpr (K == 1) {
          half *s_x = reinterpret_cast<half *>(s_x2);
          int idx = c - pipeline_id * page_size_fp32 * 2;
          dense_matrix_runner.accs[0] +=
              __half2float(v) * __half2float(s_x[idx]);
        } else {
#pragma loop unroll
          for (int j = 0; j < K; j += 2) {
            int idx = c * K / 2 + j / 2 - pipeline_id * page_size_fp32;
            float2 x2_fp32 = __half22float2(s_x2[idx]);
            dense_matrix_runner.accs[j] += v_fp32 * x2_fp32.x;
            dense_matrix_runner.accs[j + 1] += v_fp32 * x2_fp32.y;
          }
        }
      }
    }

    __syncthreads();
  }

  auto addr = tile_row_id * BETA1 + threadIdx.x;
  if constexpr (K == 1) {
    auto other =
        __shfl_down_sync(HALF_MASK, dense_matrix_runner.accs[0], BETA1);
    dense_matrix_runner.accs[0] =
        add_and_accum(other, dense_matrix_runner.accs[0]);

    auto *s_fp32_buff = reinterpret_cast<float *>(s_x2);

    u32 subwarp_id = threadIdx.x / WARP_SIZE;
    if (subwarp_id >= 1 && threadIdx.x % WARP_SIZE < BETA1) {
      s_fp32_buff[(subwarp_id - 1) * BETA1 + threadIdx.x % WARP_SIZE] =
          dense_matrix_runner.accs[0];
    }

    __syncthreads();

    if (!subtile_id && threadIdx.x < BETA1) {
      for (int i = 0; i < WARP_COUNT - 1; i++) {
        dense_matrix_runner.accs[0] += s_fp32_buff[i * BETA1 + threadIdx.x];
      }
    }

  } else {
    const u32 subwarp_id = threadIdx.x / WARP_SIZE;
    const u32 lane_id = threadIdx.x & 0x1f;
    for (int i = 0; i < K; i++) {
      auto *s_fp32_buff = reinterpret_cast<float *>(s_x2);
      float acc = dense_matrix_runner.accs[i];
      auto other = __shfl_down_sync(HALF_MASK, acc, BETA1);
      acc = add_and_accum(other, acc);

      if (subwarp_id >= 1 && lane_id < BETA1) {
        s_fp32_buff[(subwarp_id - 1) * WARP_SIZE + lane_id] = acc;
      }

      __syncthreads();

      if constexpr (THREAD_COUNT > BETA1) {
        if (!subtile_id && thread_xy < BETA1) {
          for (int j = 0; j < WARP_COUNT - 1; j++) {
            acc += s_fp32_buff[j * WARP_SIZE + threadIdx.x];
          }
        }
      }

      dense_matrix_runner.accs[i] = acc;
    }
  }

  if (threadIdx.x < BETA1) {
    auto addr = tile_row_id * BETA1 + threadIdx.x;

    if constexpr (K == 1) {
      y_fp16[K * addr] = __float2half(dense_matrix_runner.accs[0]);
    } else {
#pragma loop unroll
      for (int i = 0; i < K; i += 2) {
        y_fp32[(K * addr + i) / 2] =
            make_half2(__float2half_rd(dense_matrix_runner.accs[i]),
                       __float2half_rd(dense_matrix_runner.accs[i + 1]));
      }
    }
  }
}

template <class T> __device__ __host__ const T &__min(const T &a, const T &b) {
  return (b < a) ? b : a;
}

template <class T> __device__ __host__ const T &__max(const T &a, const T &b) {
  return (b < a) ? a : b;
}
union Features {
  uint32_t _;

  struct {
    uint32_t is_fp32 : 1;
    uint32_t dense_only : 1;
    uint32_t naive_sparse : 1;
    uint32_t torch : 1;
    uint32_t is_async : 1;
    uint32_t shared_sparse : 1;
    uint32_t single_sparse : 1;
    uint32_t cusparse : 1;
    uint32_t fused_sparse : 1;
    uint32_t shared_sparse_baseline : 1;
    uint32_t shared_mixture : 1;
    uint32_t rest : 21;
  } flags;
};

int spqr_matvec(
    // W and meta
    int bits, int m, int n,
    // Quantization
    int beta1, int beta2, const void *raw_in_order, const void *raw_dense_data,
    // 32-bit
    int row_offsets_len, void *row_offsets,
    // 16-bit
    void *col_vals, int nnz,

    // 16-bit
    // Input
    void *X,
    // Output
    void *y, cudaStream_t stream, void *measurements, uint32_t feature_flag) {
  Timer *timer{};
  if (measurements) {
    timer = new Timer(stream);
    timer->start();
  }

  if (m == 0 || n == 0) {
    return 0;
  }

  Features features{._ = feature_flag};

  const auto *raw_data_ptr = (const u64 *)raw_dense_data;
  const half *X_ptr = (const half *)X;
  const int *row_offsets_ptr = (const int *)row_offsets;
  half *y_ptr = (half *)y;
  const auto *col_vals_ptr = (const u32 *)col_vals;
  const auto *order_ptr = (const uint16_t *)raw_in_order;

  int ret = 0;

  bool is_csr = m + 1 == row_offsets_len;

  if (order_ptr == nullptr) {
    if (is_csr) {
      if (m % 16 == 0 && n % 512 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, true);
      } else if (m % 16 == 0 && n % 256 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, true);
      } else if (m % 16 == 0 && n % 128 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 8, 1, true);
      } else if (m % 16 == 0 && n % 64 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 4, 1, true);
      } else if (m % 16 == 0 && n % 32 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 2, 1, true);
      } else {
        CALL_MATVEC(spqr_quantized_matvec, 1, 1, 1, true);
      }
    } else {
      if (m % 16 == 0 && n % 512 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, false);
      } else if (m % 16 == 0 && n % 256 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 16, 1, false);
      } else if (m % 16 == 0 && n % 128 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 8, 2, false);
      } else if (m % 16 == 0 && n % 64 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 4, 1, false);
      } else if (m % 16 == 0 && n % 32 == 0) {
        CALL_MATVEC(spqr_quantized_matvec, 1, 2, 1, false);
      } else {
        CALL_MATVEC(spqr_quantized_matvec, 1, 1, 1, false);
      }
    }
  } else {
    if (is_csr) {
      if (m % 16 == 0 && n % 512 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, true);
      } else if (m % 16 == 0 && n % 256 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, true);
      } else if (m % 16 == 0 && n % 128 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 8, 1, true);
      } else if (m % 16 == 0 && n % 64 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 4, 1, true);
      } else if (m % 16 == 0 && n % 32 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 2, 1, true);
      } else {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 1, 1, true);
      }
    } else {
      if (m % 16 == 0 && n % 512 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, false);
      } else if (m % 16 == 0 && n % 256 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 16, 1, false);
      } else if (m % 16 == 0 && n % 128 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 8, 2, false);
      } else if (m % 16 == 0 && n % 64 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 4, 1, false);
      } else if (m % 16 == 0 && n % 32 == 0) {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 2, 1, false);
      } else {
        CALL_FUSED(spqr_quantized_matvec_fused, 1, 1, 1, false);
      }
    }
  }

  if (measurements) {
    static_cast<float *>(measurements)[0] = timer->end();
    delete timer;
  } else if (!features.flags.is_async) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  return ret;
}

#define CALL_MATVEC_V2

#define CALL_BATCHED_K(K)                                                      \
  if (is_csr) {                                                                \
    if (n % (TILE_COUNT * 16) == 0) {                                          \
      CALL_BATCHED_V2(spqr_quantized_matvec_batched_v2, 1, TILE_COUNT, 1,      \
                      true, K);                                                \
    } else {                                                                   \
      CALL_BATCHED_V2(spqr_quantized_matvec_batched_v2, 1, 1, 1, true, K);     \
    }                                                                          \
  } else {                                                                     \
    if (n % (TILE_COUNT * 16) == 0) {                                          \
      CALL_BATCHED_V2(spqr_quantized_matvec_batched_v2, 1, TILE_COUNT, 1,      \
                      false, K);                                               \
    } else {                                                                   \
      CALL_BATCHED_V2(spqr_quantized_matvec_batched_v2, 1, 1, 1, false, K);    \
    }                                                                          \
  }

int spqr_matvec_batched(
    // W and meta
    int bits, int m, int n, int k,
    // Quantization
    int beta1, int beta2, const void *raw_in_order, const void *raw_dense_data,
    // 32-bit
    int row_offsets_len, void *row_offsets,
    // 16-bit
    void *col_vals, int nnz,
    // Input
    void *X,
    // Output
    void *y, cudaStream_t stream, void *measurements, uint32_t feature_flag) {
  Timer *timer{};
  if (measurements) {
    timer = new Timer(stream);
    timer->start();
  }
  Features features{._ = feature_flag};

  const auto *raw_data_ptr = (const u64 *)raw_dense_data;
  const half *X_ptr = (const half *)X;
  const int *row_offsets_ptr = (const int *)row_offsets;
  half *y_ptr = (half *)y;
  const auto *col_vals_ptr = (const u32 *)col_vals;
  const auto *order_ptr = (const uint16_t *)raw_in_order;

  int ret = 0;
  bool is_csr = m + 1 == row_offsets_len;

  bool needs_fusion = order_ptr == nullptr;

  static constexpr int TILE_COUNT = 16;

  if (k == 1) {
    CALL_BATCHED_K(1)
  } else if (k == 2) {
    CALL_BATCHED_K(2)
  } else if (k == 4) {
    CALL_BATCHED_K(4)
  } else if (k == 8) {
    CALL_BATCHED_K(8)
  }

  if (measurements) {
    static_cast<float *>(measurements)[0] = timer->end();
    delete timer;
  } else if (!features.flags.is_async) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  return ret;
}
